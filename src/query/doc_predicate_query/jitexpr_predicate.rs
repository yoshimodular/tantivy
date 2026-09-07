use std::collections::HashMap;

use columnar::{ColumnType, DynamicColumn};
use jitexpr::ast::{infer_types_with_target, InferredTypeSet, TypeError, UntypedExpr};
use jitexpr::compile::{compile, CompiledFnCtx};
use jitexpr::types::{VarType, VariableValue};
use smallvec::SmallVec;

use super::{DocPredicate, SegmentDocPredicate};
use crate::index::SegmentReader;
use crate::{DocId, TantivyError};

/// A [`DocPredicate`] that evaluates a boolean JIT expression against fast fields.
///
/// Requires the `jitexpr` feature. Variable names are resolved as fast-field names
/// for each segment, supporting boolean, numeric, and string columns. Missing or
/// incompatible columns are left unbound, so the compiler treats them as `None`.
/// For bound columns, multivalued documents contribute their first value, and a
/// document missing any input is skipped before evaluating the expression.
/// Only a present `true` result matches.
///
/// ```
/// use tantivy::jitexpr::ast::deserialize;
/// use tantivy::query::doc_predicate_query::{DocPredicateQuery, JitExprPredicate};
///
/// let expression = deserialize("(EQ (ADD price 1u64) 10u64)").unwrap();
/// let query: DocPredicateQuery = JitExprPredicate::new(expression).unwrap().into();
/// ```
#[derive(Clone, Debug)]
pub struct JitExprPredicate {
    expression: UntypedExpr,
    inferred_inputs: Vec<(String, InferredTypeSet)>,
}

impl JitExprPredicate {
    /// Creates a predicate after inferring its inputs and requiring a boolean result.
    pub fn new(expression: UntypedExpr) -> Result<Self, TypeError> {
        let inferred_types = infer_types_with_target(&expression, InferredTypeSet::BOOLEAN)?;
        let mut inferred_inputs: Vec<(String, InferredTypeSet)> = inferred_types
            .into_iter()
            .map(|(name, types)| (name.to_string(), types))
            .collect();
        inferred_inputs.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        Ok(Self {
            expression,
            inferred_inputs,
        })
    }

    /// Returns the expression evaluated by this predicate.
    pub fn expression(&self) -> &UntypedExpr {
        &self.expression
    }
}

impl DocPredicate for JitExprPredicate {
    type SegmentDocPredicate = JitExprSegmentPredicate;

    fn doc_predicate(
        &self,
        segment_reader: &SegmentReader,
    ) -> crate::Result<JitExprSegmentPredicate> {
        let mut variable_types = HashMap::with_capacity(self.inferred_inputs.len());
        let mut opened_columns = HashMap::with_capacity(self.inferred_inputs.len());
        for (name, accepted_types) in &self.inferred_inputs {
            let Some(column) = open_input_column(segment_reader, name, *accepted_types)? else {
                continue;
            };
            let var_type = var_type_for_column_type(column.column_type())
                .expect("open_input_column only returns supported column types");
            variable_types.insert(name.as_str(), var_type);
            opened_columns.insert(name.as_str(), column);
        }

        let compiled = compile(&self.expression, &variable_types).map_err(|error| {
            TantivyError::InvalidArgument(format!(
                "failed to compile JIT predicate `{}`: {error}",
                self.expression
            ))
        })?;
        if compiled.result_type() != VarType::Bool {
            return Ok(JitExprSegmentPredicate { state: None });
        }

        // The compiler owns the definitive ABI order. Do not rely on inference
        // or HashMap iteration order when building the argument slots.
        let mut columns = Vec::with_capacity(compiled.inputs().len());
        for input in compiled.inputs() {
            let column = opened_columns
                .remove(input.variable_name.as_ref())
                .ok_or_else(|| {
                    TantivyError::InternalError(format!(
                        "compiled input `{}` has no corresponding fast-field column",
                        input.variable_name
                    ))
                })?;
            if var_type_for_column_type(column.column_type()) != Some(input.r#type) {
                return Err(TantivyError::InternalError(format!(
                    "compiled input `{}` expects {:?}, but its column has type {}",
                    input.variable_name,
                    input.r#type,
                    column.column_type()
                )));
            }
            columns.push(column);
        }
        // There is one reusable buffer per string column, in ABI order.
        let num_string_inputs = columns
            .iter()
            .filter(|column| matches!(column, DynamicColumn::Str(_)))
            .count();
        Ok(JitExprSegmentPredicate {
            state: Some(JitExprEvalState {
                compiled: compiled.context(),
                columns,
                string_inputs: vec![String::new(); num_string_inputs],
            }),
        })
    }
}

fn open_input_column(
    reader: &SegmentReader,
    name: &str,
    accepted_types: InferredTypeSet,
) -> crate::Result<Option<DynamicColumn>> {
    for handle in reader.fast_fields().dynamic_column_handles(name)? {
        let Some(var_type) = var_type_for_column_type(handle.column_type()) else {
            continue;
        };
        if accepted_types.contains(var_type) {
            return Ok(Some(handle.open()?));
        }
    }
    Ok(None)
}

fn var_type_for_column_type(column_type: ColumnType) -> Option<VarType> {
    match column_type {
        ColumnType::Bool => Some(VarType::Bool),
        ColumnType::I64 => Some(VarType::I64),
        ColumnType::U64 => Some(VarType::U64),
        ColumnType::F64 => Some(VarType::F64),
        ColumnType::Str => Some(VarType::Str),
        ColumnType::Bytes | ColumnType::IpAddr | ColumnType::DateTime => None,
    }
}

/// The [`SegmentDocPredicate`] produced by [`JitExprPredicate`] for one segment.
pub struct JitExprSegmentPredicate {
    // None means the expression cannot produce a boolean for this segment.
    state: Option<JitExprEvalState>,
}

struct JitExprEvalState {
    compiled: CompiledFnCtx,
    columns: Vec<DynamicColumn>,
    string_inputs: Vec<String>,
}

impl SegmentDocPredicate for JitExprSegmentPredicate {
    fn eval(&mut self, doc_id: DocId) -> bool {
        let Some(state) = &mut self.state else {
            return false;
        };
        state.eval(doc_id)
    }
}

impl JitExprEvalState {
    fn eval(&mut self, doc_id: DocId) -> bool {
        // Argument borrows last only for this evaluation: no self-referential
        // pointers survive into the next call, when string buffers may grow.
        // Small expressions keep their argument slots on the stack.
        let mut input_values: SmallVec<[VariableValue<'_>; 8]> =
            SmallVec::with_capacity(self.columns.len());
        let mut string_inputs = self.string_inputs.iter_mut();
        for column in &self.columns {
            let input = match column {
                DynamicColumn::Bool(column) => column.first(doc_id).map(VariableValue::from),
                DynamicColumn::I64(column) => column.first(doc_id).map(VariableValue::from),
                DynamicColumn::U64(column) => column.first(doc_id).map(VariableValue::from),
                DynamicColumn::F64(column) => column.first(doc_id).map(VariableValue::from),
                DynamicColumn::Str(column) => {
                    let string_input = string_inputs
                        .next()
                        .expect("every string column has a string input buffer");
                    let Some(term_ord) = column.ords().first(doc_id) else {
                        return false;
                    };
                    string_input.clear();
                    // SegmentDocPredicate::eval cannot return I/O errors;
                    // an unreadable dictionary therefore panics.
                    let found = column
                        .ord_to_str(term_ord, string_input)
                        .expect("a fast-field string dictionary became unreadable after opening");
                    if !found {
                        return false;
                    }
                    Some(VariableValue::from(string_input.as_str()))
                }
                DynamicColumn::Bytes(_) | DynamicColumn::IpAddr(_) | DynamicColumn::DateTime(_) => {
                    unreachable!("unsupported columns are filtered before compilation")
                }
            };
            let Some(input) = input else {
                return false;
            };
            input_values.push(input);
        }
        // SAFETY: Columns follow compiled.inputs() and their types were checked
        // during setup. Each slot uses the matching union arm. String buffers
        // remain borrowed, and cannot be mutated, until this call finishes.
        let result = unsafe { self.compiled.call(&input_values) };
        // SAFETY: Setup checked that the compiled result type is boolean.
        (unsafe { result.as_bool() }) == Some(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::Count;
    use crate::query::doc_predicate_query::DocPredicateQuery;
    use crate::query::{EnableScoring, Query};
    use crate::schema::{Schema, FAST, STRING};
    use crate::{Index, TERMINATED};

    fn create_index() -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let number = schema_builder.add_u64_field("number", FAST);
        let flag = schema_builder.add_bool_field("flag", FAST);
        let label = schema_builder.add_text_field("label", STRING | FAST);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_for_tests()?;
        writer.add_document(doc!(number => 1u64, flag => true, label => "one"))?;
        writer.add_document(doc!(number => 2u64, flag => false, label => "two"))?;
        writer.add_document(doc!(number => 3u64, flag => true, label => "three"))?;
        writer.add_document(doc!(number => 4u64))?;
        writer.commit()?;
        Ok(index)
    }

    fn query(expression: &str) -> DocPredicateQuery {
        JitExprPredicate::new(jitexpr::ast::deserialize(expression).unwrap())
            .unwrap()
            .into()
    }

    #[test]
    fn test_constructor_requires_boolean_expression() {
        let expression = jitexpr::ast::deserialize("(ADD number 1u64)").unwrap();
        assert!(JitExprPredicate::new(expression).is_err());
    }

    #[test]
    fn test_numeric_predicate_query() -> crate::Result<()> {
        let index = create_index()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.search(&query("(EQ number 2i64)"), &Count)?, 1);
        assert_eq!(
            searcher.search(&query("(EQ (ADD number 1u64) 3u64)"), &Count)?,
            1
        );
        Ok(())
    }

    #[test]
    fn test_boolean_and_string_inputs_follow_compiled_order() -> crate::Result<()> {
        let index = create_index()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.search(&query("flag"), &Count)?, 2);
        assert_eq!(searcher.search(&query(r#"(EQ label "three")"#), &Count)?, 1);
        // Inference sorts names, but the ABI follows expression order: label, flag.
        assert_eq!(
            searcher.search(&query(r#"(EQ (EQ label "three") flag)"#), &Count)?,
            2
        );
        Ok(())
    }

    #[test]
    fn test_constant_predicates() -> crate::Result<()> {
        let index = create_index()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.search(&query("true"), &Count)?, 4);
        assert_eq!(searcher.search(&query("false"), &Count)?, 0);
        Ok(())
    }

    #[test]
    fn test_missing_and_incompatible_columns() -> crate::Result<()> {
        let index = create_index()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.search(&query("missing"), &Count)?, 0);
        assert_eq!(
            searcher.search(&query("(EQ (ADD label 1i64) 2i64)"), &Count)?,
            0
        );
        // A column missing from the segment is compiled as None.
        assert_eq!(searcher.search(&query("(IS_NULL missing)"), &Count)?, 4);
        // Missing values in a bound column skip the document before evaluation.
        assert_eq!(searcher.search(&query("(IS_NULL flag)"), &Count)?, 0);
        Ok(())
    }

    #[test]
    fn test_signed_and_float_columns() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let signed = schema_builder.add_i64_field("signed", FAST);
        let float = schema_builder.add_f64_field("float", FAST);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_for_tests()?;
        writer.add_document(doc!(signed => -2i64, float => 1.5f64))?;
        writer.add_document(doc!(signed => 3i64, float => 2.5f64))?;
        writer.add_document(doc!())?;
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.search(&query("(EQ signed -2i64)"), &Count)?, 1);
        assert_eq!(searcher.search(&query("(EQ float 1.5f64)"), &Count)?, 1);
        Ok(())
    }

    #[test]
    fn test_multivalued_columns_use_first_value() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let number = schema_builder.add_u64_field("number", FAST);
        let label = schema_builder.add_text_field("label", STRING | FAST);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_for_tests()?;
        writer.add_document(doc!(number => 1u64, number => 2u64,
                                 label => "first", label => "second"))?;
        writer.add_document(doc!(number => 2u64, number => 1u64,
                                 label => "second", label => "first"))?;
        writer.add_document(doc!())?;
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.search(&query("(EQ number 1u64)"), &Count)?, 1);
        assert_eq!(searcher.search(&query(r#"(EQ label "first")"#), &Count)?, 1);
        Ok(())
    }

    #[test]
    fn test_string_buffers_are_reused_across_evaluations() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let left = schema_builder.add_text_field("left", STRING | FAST);
        let right = schema_builder.add_text_field("right", STRING | FAST);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_for_tests()?;
        let long_string = "long string ".repeat(1000);
        writer.add_document(doc!(left => "short", right => "SHORT"))?;
        writer
            .add_document(doc!(left => long_string.clone(), right => long_string.to_uppercase()))?;
        writer.add_document(doc!(left => "", right => ""))?;
        writer.add_document(doc!(left => "missing right"))?;
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        // This also exercises the JIT context's arena-backed string results.
        let predicate =
            JitExprPredicate::new(jitexpr::ast::deserialize("(EQ right (UPPER left))").unwrap())
                .unwrap();
        let mut segment_predicate = predicate.doc_predicate(searcher.segment_reader(0))?;
        for _ in 0..3 {
            assert!(segment_predicate.eval(0));
            assert!(segment_predicate.eval(1));
            assert!(segment_predicate.eval(2));
            assert!(!segment_predicate.eval(3));
        }
        Ok(())
    }

    #[test]
    fn test_more_than_eight_inputs() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let mut document = crate::TantivyDocument::default();
        let mut sum = "0u64".to_string();
        for i in 0..9 {
            let name = format!("input_{i}");
            let field = schema_builder.add_u64_field(&name, FAST);
            document.add_u64(field, i + 1);
            sum = format!("(ADD {sum} {name})");
        }
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_for_tests()?;
        writer.add_document(document)?;
        writer.add_document(doc!())?;
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        let predicate =
            JitExprPredicate::new(jitexpr::ast::deserialize(&format!("(EQ {sum} 45u64)")).unwrap())
                .unwrap();
        let mut segment_predicate = predicate.doc_predicate(searcher.segment_reader(0))?;
        assert_eq!(segment_predicate.state.as_ref().unwrap().columns.len(), 9);
        assert!(segment_predicate.eval(0));
        assert!(!segment_predicate.eval(1));
        Ok(())
    }

    #[test]
    fn test_query_scorer_and_explanation() -> crate::Result<()> {
        let index = create_index()?;
        let searcher = index.reader()?.searcher();
        let query = query("flag");
        let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let reader = searcher.segment_reader(0);
        assert!(weight.explain(reader, 0).is_ok());
        assert!(weight.explain(reader, 1).is_err());
        let mut scorer = weight.scorer(reader, 2.0)?;
        assert_eq!(scorer.doc(), 0);
        assert_eq!(scorer.score(), 2.0);
        assert_eq!(scorer.advance(), 2);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }
}
