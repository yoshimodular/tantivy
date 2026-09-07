use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, Weak};

use common::StableDeref;
use fs4::FileExt;
#[cfg(all(feature = "mmap", unix))]
pub use memmap2::Advice;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use crate::core::META_FILEPATH;
use crate::directory::error::{
    DeleteError, LockError, OpenDirectoryError, OpenReadError, OpenWriteError,
};
use crate::directory::file_watcher::FileWatcher;
use crate::directory::{
    AntiCallToken, Directory, DirectoryLock, FileHandle, Lock, OwnedBytes, TerminatingWrite,
    WatchCallback, WatchHandle, WritePtr,
};

pub type ArcBytes = Arc<dyn Deref<Target = [u8]> + Send + Sync + 'static>;
pub type WeakArcBytes = Weak<dyn Deref<Target = [u8]> + Send + Sync + 'static>;

/// Create a default io error given a string.
pub(crate) fn make_io_err(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::Other, msg)
}

/// Returns `None` iff the file exists, can be read, but is empty (and hence
/// cannot be mmapped)
fn open_mmap(full_path: &Path) -> Result<Option<Mmap>, OpenReadError> {
    let file = File::open(full_path).map_err(|io_err| {
        if io_err.kind() == io::ErrorKind::NotFound {
            OpenReadError::FileDoesNotExist(full_path.to_path_buf())
        } else {
            OpenReadError::wrap_io_error(io_err, full_path.to_path_buf())
        }
    })?;

    let meta_data = file
        .metadata()
        .map_err(|io_err| OpenReadError::wrap_io_error(io_err, full_path.to_owned()))?;
    if meta_data.len() == 0 {
        // if the file size is 0, it will not be possible
        // to mmap the file, so we return None
        // instead.
        return Ok(None);
    }
    let mmap_opt: Option<memmap2::Mmap> = unsafe {
        memmap2::Mmap::map(&file)
            .map(Some)
            .map_err(|io_err| OpenReadError::wrap_io_error(io_err, full_path.to_path_buf()))
    }?;

    Ok(mmap_opt)
}

fn mmap_file(file: File, logical_path: &Path) -> Result<Option<Mmap>, OpenReadError> {
    let meta_data = file
        .metadata()
        .map_err(|io_err| OpenReadError::wrap_io_error(io_err, logical_path.to_owned()))?;
    if meta_data.len() == 0 {
        return Ok(None);
    }
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|io_err| OpenReadError::wrap_io_error(io_err, logical_path.to_owned()))?;
    Ok(Some(mmap))
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct CacheCounters {
    /// Number of time the cache prevents to call `mmap`
    pub hit: usize,
    /// Number of time tantivy had to call `mmap`
    /// as no entry was in the cache.
    pub miss: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheInfo {
    pub counters: CacheCounters,
    pub mmapped: Vec<PathBuf>,
}

struct MmapCache {
    counters: CacheCounters,
    cache: HashMap<PathBuf, WeakArcBytes>,
    #[cfg(unix)]
    madvice_opt: Option<Advice>,
}

impl MmapCache {
    fn new() -> MmapCache {
        MmapCache {
            counters: CacheCounters::default(),
            cache: HashMap::default(),
            #[cfg(unix)]
            madvice_opt: None,
        }
    }

    #[cfg(unix)]
    fn set_advice(&mut self, madvice: Advice) {
        self.madvice_opt = Some(madvice);
    }

    fn get_info(&self) -> CacheInfo {
        let paths: Vec<PathBuf> = self.cache.keys().cloned().collect();
        CacheInfo {
            counters: self.counters.clone(),
            mmapped: paths,
        }
    }

    fn remove_weak_ref(&mut self) {
        let keys_to_remove: Vec<PathBuf> = self
            .cache
            .iter()
            .filter(|(_, mmap_weakref)| mmap_weakref.upgrade().is_none())
            .map(|(key, _)| key.clone())
            .collect();
        for key in keys_to_remove {
            self.cache.remove(&key);
        }
    }

    fn open_mmap_impl(&self, full_path: &Path) -> Result<Option<Mmap>, OpenReadError> {
        let mmap_opt = open_mmap(full_path)?;
        #[cfg(unix)]
        if let (Some(mmap), Some(madvice)) = (mmap_opt.as_ref(), self.madvice_opt) {
            // We ignore madvise errors.
            let _ = mmap.advise(madvice);
        }
        Ok(mmap_opt)
    }

    // Returns None if the file exists but as a len of 0 (and hence is not mmappable).
    fn get_mmap(&mut self, full_path: &Path) -> Result<Option<ArcBytes>, OpenReadError> {
        if let Some(mmap_weak) = self.cache.get(full_path) {
            if let Some(mmap_arc) = mmap_weak.upgrade() {
                self.counters.hit += 1;
                return Ok(Some(mmap_arc));
            }
        }
        self.cache.remove(full_path);
        self.counters.miss += 1;
        let mmap_opt = self.open_mmap_impl(full_path)?;
        Ok(mmap_opt.map(|mmap| {
            let mmap_arc: ArcBytes = Arc::new(mmap);
            let mmap_weak = Arc::downgrade(&mmap_arc);
            self.cache.insert(full_path.to_owned(), mmap_weak);
            mmap_arc
        }))
    }

    #[cfg(unix)]
    fn get_mmap_from_file(
        &mut self,
        logical_path: &Path,
        file: File,
    ) -> Result<Option<ArcBytes>, OpenReadError> {
        if let Some(mmap_weak) = self.cache.get(logical_path) {
            if let Some(mmap_arc) = mmap_weak.upgrade() {
                self.counters.hit += 1;
                return Ok(Some(mmap_arc));
            }
        }
        self.cache.remove(logical_path);
        self.counters.miss += 1;
        let mmap_opt = mmap_file(file, logical_path)?;
        Ok(mmap_opt.map(|mmap| {
            let mmap_arc: ArcBytes = Arc::new(mmap);
            self.cache
                .insert(logical_path.to_owned(), Arc::downgrade(&mmap_arc));
            mmap_arc
        }))
    }
}

/// Directory storing data in files, read via mmap.
///
/// The Mmap object are cached to limit the
/// system calls.
///
/// In the `MmapDirectory`, locks are implemented using the `fs2` crate definition of locks.
///
/// On MacOS & linux, it relies on `flock` (aka `BSD Lock`). These locks solve most of the
/// problems related to POSIX Locks, but may their contract may not be respected on `NFS`
/// depending on the implementation.
///
/// On Windows the semantics are again different.
#[derive(Clone)]
pub struct MmapDirectory {
    inner: Arc<MmapDirectoryInner>,
}

struct MmapDirectoryInner {
    root: DirectoryRoot,
    mmap_cache: RwLock<MmapCache>,
    _temp_directory: Option<TempDir>,
    watcher: FileWatcher,
}

#[derive(Clone)]
enum DirectoryRoot {
    Path(PathBuf),
    #[cfg(unix)]
    Capability(Arc<File>),
}

#[cfg(unix)]
fn capability_name(path: &Path) -> io::Result<std::ffi::CString> {
    use std::os::unix::ffi::OsStrExt;
    let name = match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) if parent.as_os_str().is_empty() => name,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "capability directory only accepts flat child names",
            ));
        }
    };
    std::ffi::CString::new(name.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL in child name"))
}

#[cfg(unix)]
fn capability_open(
    dir: &File,
    path: &Path,
    flags: libc::c_int,
    mode: libc::mode_t,
) -> io::Result<File> {
    use std::os::unix::io::{AsRawFd, FromRawFd};
    let name = capability_name(path)?;
    let fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            mode as libc::c_uint,
        )
    };
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}

#[cfg(unix)]
fn capability_delete(dir: &File, path: &Path) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let name = capability_name(path)?;
    let rc = unsafe { libc::unlinkat(dir.as_raw_fd(), name.as_ptr(), 0) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn capability_rename(dir: &File, from: &Path, to: &Path) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let from = capability_name(from)?;
    let to = capability_name(to)?;
    let rc =
        unsafe { libc::renameat(dir.as_raw_fd(), from.as_ptr(), dir.as_raw_fd(), to.as_ptr()) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn capability_atomic_write(dir: &File, path: &Path, content: &[u8]) -> io::Result<()> {
    let temporary = PathBuf::from(format!(".tantivy-atomic-{}", uuid::Uuid::new_v4()));
    let mut file = capability_open(
        dir,
        &temporary,
        libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        0o600,
    )?;
    let result = (|| {
        file.write_all(content)?;
        file.flush()?;
        file.sync_data()?;
        capability_rename(dir, &temporary, path)?;
        dir.sync_data()
    })();
    if result.is_err() {
        let _ = capability_delete(dir, &temporary);
    }
    result
}

impl MmapDirectoryInner {
    fn new(root_path: PathBuf, temp_directory: Option<TempDir>) -> MmapDirectoryInner {
        MmapDirectoryInner {
            mmap_cache: RwLock::new(MmapCache::new()),
            _temp_directory: temp_directory,
            watcher: FileWatcher::new(&root_path.join(*META_FILEPATH)),
            root: DirectoryRoot::Path(root_path),
        }
    }

    #[cfg(unix)]
    fn from_dir(dir: File) -> io::Result<MmapDirectoryInner> {
        if !dir.metadata()?.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "capability is not a directory",
            ));
        }
        let dir = Arc::new(dir);
        Ok(MmapDirectoryInner {
            mmap_cache: RwLock::new(MmapCache::new()),
            _temp_directory: None,
            watcher: FileWatcher::new_from_dir(dir.clone(), META_FILEPATH.as_ref()),
            root: DirectoryRoot::Capability(dir),
        })
    }

    fn watch(&self, callback: WatchCallback) -> WatchHandle {
        self.watcher.watch(callback)
    }
}

impl fmt::Debug for MmapDirectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.root {
            DirectoryRoot::Path(path) => write!(f, "MmapDirectory({path:?})"),
            #[cfg(unix)]
            DirectoryRoot::Capability(_) => write!(f, "MmapDirectory(<directory capability>)"),
        }
    }
}

impl MmapDirectory {
    fn new(root_path: PathBuf, temp_directory: Option<TempDir>) -> MmapDirectory {
        let inner = MmapDirectoryInner::new(root_path, temp_directory);
        MmapDirectory {
            inner: Arc::new(inner),
        }
    }

    /// Creates a new MmapDirectory in a temporary directory.
    ///
    /// This is mostly useful to test the MmapDirectory itself.
    /// For your unit tests, prefer the RamDirectory.
    pub fn create_from_tempdir() -> Result<MmapDirectory, OpenDirectoryError> {
        let tempdir = TempDir::new()
            .map_err(|io_err| OpenDirectoryError::FailedToCreateTempDir(Arc::new(io_err)))?;
        Ok(MmapDirectory::new(
            tempdir.path().to_path_buf(),
            Some(tempdir),
        ))
    }

    /// Opens a MmapDirectory in a directory, with a given access pattern.
    ///
    /// This is only supported on unix platforms.
    #[cfg(unix)]
    pub fn open_with_madvice(
        directory_path: impl AsRef<Path>,
        madvice: Advice,
    ) -> Result<MmapDirectory, OpenDirectoryError> {
        let dir = Self::open_impl_to_avoid_monomorphization(directory_path.as_ref())?;
        dir.inner.mmap_cache.write().unwrap().set_advice(madvice);
        Ok(dir)
    }

    /// Opens a MmapDirectory in a directory.
    ///
    /// Returns an error if the `directory_path` does not
    /// exist or if it is not a directory.
    pub fn open(directory_path: impl AsRef<Path>) -> Result<MmapDirectory, OpenDirectoryError> {
        Self::open_impl_to_avoid_monomorphization(directory_path.as_ref())
    }

    /// Opens a directory from an already validated directory descriptor.
    ///
    /// Unlike [`MmapDirectory::open`], every later operation is relative to the
    /// captured descriptor and refuses symlink children. Renaming or replacing
    /// the pathname used to obtain `directory` therefore cannot retarget this
    /// instance to a different index.
    #[cfg(unix)]
    pub fn open_from_dir(directory: File) -> Result<MmapDirectory, OpenDirectoryError> {
        let inner = MmapDirectoryInner::from_dir(directory).map_err(|error| {
            OpenDirectoryError::wrap_io_error(error, PathBuf::from("<directory capability>"))
        })?;
        Ok(MmapDirectory {
            inner: Arc::new(inner),
        })
    }

    #[inline(never)]
    fn open_impl_to_avoid_monomorphization(
        directory_path: &Path,
    ) -> Result<MmapDirectory, OpenDirectoryError> {
        if !directory_path.exists() {
            return Err(OpenDirectoryError::DoesNotExist(PathBuf::from(
                directory_path,
            )));
        }
        #[allow(clippy::bind_instead_of_map)]
        let canonical_path: PathBuf = directory_path.canonicalize().or_else(|io_err| {
            let directory_path = directory_path.to_owned();

            #[cfg(windows)]
            {
                // `canonicalize` returns "Incorrect function" (error code 1)
                // for virtual drives (network drives, ramdisk, etc.).
                if io_err.raw_os_error() == Some(1) && directory_path.exists() {
                    // Should call `std::path::absolute` when it is stabilised.
                    return Ok(directory_path);
                }
            }

            Err(OpenDirectoryError::wrap_io_error(io_err, directory_path))
        })?;
        if !canonical_path.is_dir() {
            return Err(OpenDirectoryError::NotADirectory(PathBuf::from(
                directory_path,
            )));
        }
        Ok(MmapDirectory::new(canonical_path, None))
    }

    /// Joins a relative_path to the directory `root_path`
    /// to create a proper complete `filepath`.
    fn resolve_path(&self, relative_path: &Path) -> PathBuf {
        match &self.inner.root {
            DirectoryRoot::Path(root) => root.join(relative_path),
            #[cfg(unix)]
            DirectoryRoot::Capability(_) => relative_path.to_owned(),
        }
    }

    /// Returns some statistical information
    /// about the Mmap cache.
    ///
    /// The `MmapDirectory` embeds a `MmapDirectory`
    /// to avoid multiplying the `mmap` system calls.
    pub fn get_cache_info(&self) -> CacheInfo {
        self.inner
            .mmap_cache
            .write()
            .expect("mmap cache lock is poisoned")
            .remove_weak_ref();
        self.inner
            .mmap_cache
            .read()
            .expect("Mmap cache lock is poisoned.")
            .get_info()
    }
}

/// We rely on fs2 for file locking. On Windows & MacOS this
/// uses BSD locks (`flock`). The lock is actually released when
/// the `File` object is dropped and its associated file descriptor
/// is closed.
struct ReleaseLockFile {
    _file: File,
    path: PathBuf,
}

impl Drop for ReleaseLockFile {
    fn drop(&mut self) {
        debug!("Releasing lock {:?}", self.path);
    }
}

/// This Write wraps a File, but has the specificity of
/// call `sync_all` on flush.
struct SafeFileWriter(File);

impl SafeFileWriter {
    fn new(file: File) -> SafeFileWriter {
        SafeFileWriter(file)
    }
}

impl Write for SafeFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl TerminatingWrite for SafeFileWriter {
    fn terminate_ref(&mut self, _: AntiCallToken) -> io::Result<()> {
        self.0.flush()?;
        self.0.sync_data()?;
        Ok(())
    }
}

#[derive(Clone)]
struct MmapArc(Arc<dyn Deref<Target = [u8]> + Send + Sync>);

impl Deref for MmapArc {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.0.deref()
    }
}
unsafe impl StableDeref for MmapArc {}

/// Writes a file in an atomic manner.
pub(crate) fn atomic_write(path: &Path, content: &[u8]) -> io::Result<()> {
    // We create the temporary file in the same directory as the target file.
    // Indeed the canonical temp directory and the target file might sit in different
    // filesystem, in which case the atomic write may actually not work.
    let parent_path = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Path {:?} does not have parent directory.",
        )
    })?;
    let mut tempfile = tempfile::Builder::new().tempfile_in(parent_path)?;
    tempfile.write_all(content)?;
    tempfile.flush()?;
    tempfile.as_file_mut().sync_data()?;
    tempfile.into_temp_path().persist(path)?;
    Ok(())
}

impl Directory for MmapDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        debug!("Open Read {:?}", path);
        let full_path = self.resolve_path(path);

        let mut mmap_cache = self.inner.mmap_cache.write().map_err(|_| {
            let msg = format!("Failed to acquired write lock on mmap cache while reading {path:?}");
            let io_err = make_io_err(msg);
            OpenReadError::wrap_io_error(io_err, path.to_path_buf())
        })?;

        let mmap = match &self.inner.root {
            DirectoryRoot::Path(_) => mmap_cache.get_mmap(&full_path)?,
            #[cfg(unix)]
            DirectoryRoot::Capability(dir) => {
                let file = capability_open(dir, path, libc::O_RDONLY, 0).map_err(|io_err| {
                    if io_err.kind() == io::ErrorKind::NotFound {
                        OpenReadError::FileDoesNotExist(path.to_owned())
                    } else {
                        OpenReadError::wrap_io_error(io_err, path.to_owned())
                    }
                })?;
                mmap_cache.get_mmap_from_file(path, file)?
            }
        };
        let owned_bytes = mmap
            .map(|mmap_arc| {
                let mmap_arc_obj = MmapArc(mmap_arc);
                OwnedBytes::new(mmap_arc_obj)
            })
            .unwrap_or_else(OwnedBytes::empty);

        Ok(Arc::new(owned_bytes))
    }

    /// Any entry associated with the path in the mmap will be
    /// removed before the file is deleted.
    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        let full_path = self.resolve_path(path);
        let result = match &self.inner.root {
            DirectoryRoot::Path(_) => fs::remove_file(full_path),
            #[cfg(unix)]
            DirectoryRoot::Capability(dir) => capability_delete(dir, path),
        };
        result.map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                DeleteError::FileDoesNotExist(path.to_owned())
            } else {
                DeleteError::IoError {
                    io_error: Arc::new(e),
                    filepath: path.to_path_buf(),
                }
            }
        })?;
        Ok(())
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        let full_path = self.resolve_path(path);
        match &self.inner.root {
            DirectoryRoot::Path(_) => full_path
                .try_exists()
                .map_err(|io_err| OpenReadError::wrap_io_error(io_err, path.to_path_buf())),
            #[cfg(unix)]
            DirectoryRoot::Capability(dir) => match capability_open(dir, path, libc::O_RDONLY, 0) {
                Ok(_) => Ok(true),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
                Err(error) => Err(OpenReadError::wrap_io_error(error, path.to_owned())),
            },
        }
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        debug!("Open Write {:?}", path);
        let full_path = self.resolve_path(path);

        let open_res = match &self.inner.root {
            DirectoryRoot::Path(_) => OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(full_path),
            #[cfg(unix)]
            DirectoryRoot::Capability(dir) => capability_open(
                dir,
                path,
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
                0o600,
            ),
        };

        let mut file = open_res.map_err(|io_err| {
            if io_err.kind() == io::ErrorKind::AlreadyExists {
                OpenWriteError::FileAlreadyExists(path.to_path_buf())
            } else {
                OpenWriteError::wrap_io_error(io_err, path.to_path_buf())
            }
        })?;

        // making sure the file is created.
        file.flush()
            .map_err(|io_error| OpenWriteError::wrap_io_error(io_error, path.to_path_buf()))?;

        // Note we actually do not sync the parent directory here.
        //
        // A newly created file, may, in some case, be created and even flushed to disk.
        // and then lost...
        //
        // The file will only be durably written after we terminate AND
        // sync_directory() is called.

        let writer = SafeFileWriter::new(file);
        Ok(BufWriter::new(Box::new(writer)))
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let full_path = self.resolve_path(path);
        let mut buffer = Vec::new();
        let opened = match &self.inner.root {
            DirectoryRoot::Path(_) => File::open(full_path),
            #[cfg(unix)]
            DirectoryRoot::Capability(dir) => capability_open(dir, path, libc::O_RDONLY, 0),
        };
        match opened {
            Ok(mut file) => {
                file.read_to_end(&mut buffer).map_err(|io_error| {
                    OpenReadError::wrap_io_error(io_error, path.to_path_buf())
                })?;
                Ok(buffer)
            }
            Err(io_error) => {
                if io_error.kind() == io::ErrorKind::NotFound {
                    Err(OpenReadError::FileDoesNotExist(path.to_owned()))
                } else {
                    Err(OpenReadError::wrap_io_error(io_error, path.to_path_buf()))
                }
            }
        }
    }

    fn atomic_write(&self, path: &Path, content: &[u8]) -> io::Result<()> {
        debug!("Atomic Write {:?}", path);
        let full_path = self.resolve_path(path);
        match &self.inner.root {
            DirectoryRoot::Path(_) => atomic_write(&full_path, content),
            #[cfg(unix)]
            DirectoryRoot::Capability(dir) => capability_atomic_write(dir, path, content),
        }
    }

    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        let full_path = self.resolve_path(&lock.filepath);
        // We make sure that the file exists.
        let file: File = match &self.inner.root {
            DirectoryRoot::Path(_) => OpenOptions::new()
                .write(true)
                .create(true) //< if the file does not exist yet, create it.
                .truncate(false)
                .open(full_path),
            #[cfg(unix)]
            DirectoryRoot::Capability(dir) => {
                capability_open(dir, &lock.filepath, libc::O_RDWR | libc::O_CREAT, 0o600)
            }
        }
        .map_err(LockError::wrap_io_error)?;
        if lock.is_blocking {
            file.lock_exclusive().map_err(LockError::wrap_io_error)?;
        } else {
            file.try_lock_exclusive().map_err(|_| LockError::LockBusy)?
        }
        // dropping the file handle will release the lock.
        Ok(DirectoryLock::from(Box::new(ReleaseLockFile {
            path: lock.filepath.clone(),
            _file: file,
        })))
    }

    fn watch(&self, watch_callback: WatchCallback) -> crate::Result<WatchHandle> {
        Ok(self.inner.watch(watch_callback))
    }

    #[cfg(windows)]
    fn sync_directory(&self) -> Result<(), io::Error> {
        // On Windows, it is not necessary to fsync the parent directory to
        // ensure that the directory entry containing the file has also reached
        // disk, and calling sync_data on a handle to directory is a no-op on
        // local disks, but will return an error on virtual drives.
        Ok(())
    }

    #[cfg(not(windows))]
    fn sync_directory(&self) -> Result<(), io::Error> {
        match &self.inner.root {
            DirectoryRoot::Path(path) => {
                let mut open_opts = OpenOptions::new();
                // Linux needs read to be set, otherwise returns EINVAL
                // write must not be set, or it fails with EISDIR
                open_opts.read(true);
                open_opts.open(path)?.sync_data()
            }
            DirectoryRoot::Capability(dir) => dir.sync_data(),
        }
    }
}

#[cfg(test)]
mod tests {

    // There are more tests in directory/mod.rs
    // The following tests are specific to the MmapDirectory

    use std::time::Duration;

    use common::HasLen;

    use super::*;
    use crate::indexer::LogMergePolicy;
    use crate::schema::{Schema, SchemaBuilder, TEXT};
    use crate::{Index, IndexSettings, IndexWriter, ReloadPolicy};

    #[test]
    fn test_open_non_existent_path() {
        assert!(MmapDirectory::open(PathBuf::from("./nowhere")).is_err());
    }

    #[test]
    fn test_open_empty() {
        // empty file is actually an edge case because those
        // cannot be mmapped.
        //
        // In that case the directory returns a SharedVecSlice.
        let mmap_directory = MmapDirectory::create_from_tempdir().unwrap();
        let path = PathBuf::from("test");
        {
            let mut w = mmap_directory.open_write(&path).unwrap();
            w.flush().unwrap();
        }
        let readonlymap = mmap_directory.open_read(&path).unwrap();
        assert_eq!(readonlymap.len(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn capability_roundtrip_survives_root_retarget_and_refuses_symlink_children() {
        use std::os::unix::fs::{symlink, OpenOptionsExt};
        use std::sync::{Arc, Barrier};

        let parent = TempDir::new().unwrap();
        let configured = parent.path().join("index");
        let captured = parent.path().join("captured");
        fs::create_dir(&configured).unwrap();
        let root = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(&configured)
            .unwrap();
        let directory = MmapDirectory::open_from_dir(root).unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let writer_directory = directory.clone();
        let writer_barrier = barrier.clone();
        let writer = std::thread::spawn(move || {
            writer_barrier.wait();
            writer_directory.atomic_write(Path::new("proof"), b"captured")
        });

        fs::rename(&configured, &captured).unwrap();
        fs::create_dir(&configured).unwrap();
        let sentinel = parent.path().join("sentinel");
        fs::write(&sentinel, b"untouched").unwrap();
        symlink(&sentinel, configured.join("proof")).unwrap();
        barrier.wait();
        writer.join().unwrap().unwrap();

        assert_eq!(fs::read(captured.join("proof")).unwrap(), b"captured");
        assert_eq!(fs::read(&sentinel).unwrap(), b"untouched");
        assert!(configured.join("proof").is_symlink());

        symlink(&sentinel, captured.join("poison")).unwrap();
        assert!(directory.atomic_read(Path::new("poison")).is_err());
        assert_eq!(fs::read(&sentinel).unwrap(), b"untouched");
    }

    #[cfg(unix)]
    #[test]
    fn capability_index_roundtrip_stays_on_captured_inode_after_barrier() {
        use std::os::unix::fs::OpenOptionsExt;
        use std::sync::{Arc, Barrier};

        let parent = TempDir::new().unwrap();
        let configured = parent.path().join("index");
        let captured = parent.path().join("captured");
        fs::create_dir(&configured).unwrap();
        let root = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(&configured)
            .unwrap();
        let directory = MmapDirectory::open_from_dir(root).unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = barrier.clone();
        let worker_directory = directory.clone();
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            let mut schema = Schema::builder();
            let text = schema.add_text_field("text", TEXT);
            let index = Index::open_or_create(worker_directory, schema.build()).unwrap();
            let mut writer = index.writer(15_000_000).unwrap();
            writer
                .add_document(crate::doc!(text => "captured"))
                .unwrap();
            writer.commit().unwrap();
            index.reader().unwrap().searcher().num_docs()
        });

        fs::rename(&configured, &captured).unwrap();
        fs::create_dir(&configured).unwrap();
        fs::write(configured.join("sentinel"), b"decoy").unwrap();
        barrier.wait();
        assert_eq!(worker.join().unwrap(), 1);
        assert!(captured.join("meta.json").is_file());
        assert!(!configured.join("meta.json").exists());
        assert_eq!(fs::read(configured.join("sentinel")).unwrap(), b"decoy");
    }

    #[test]
    fn test_cache() {
        let content = b"abc";

        // here we test if the cache releases
        // mmaps correctly.
        let mmap_directory = MmapDirectory::create_from_tempdir().unwrap();
        let num_paths = 10;
        let paths: Vec<PathBuf> = (0..num_paths)
            .map(|i| PathBuf::from(&*format!("file_{}", i)))
            .collect();
        {
            for path in &paths {
                let mut w = mmap_directory.open_write(path).unwrap();
                w.write_all(content).unwrap();
                w.flush().unwrap();
            }
        }

        let mut keep = vec![];
        for (i, path) in paths.iter().enumerate() {
            keep.push(mmap_directory.open_read(path).unwrap());
            assert_eq!(mmap_directory.get_cache_info().mmapped.len(), i + 1);
        }
        assert_eq!(mmap_directory.get_cache_info().counters.hit, 0);
        assert_eq!(mmap_directory.get_cache_info().counters.miss, 10);
        assert_eq!(mmap_directory.get_cache_info().mmapped.len(), 10);
        for path in paths.iter() {
            let _r = mmap_directory.open_read(path).unwrap();
            assert_eq!(mmap_directory.get_cache_info().mmapped.len(), num_paths);
        }
        assert_eq!(mmap_directory.get_cache_info().counters.hit, 10);
        assert_eq!(mmap_directory.get_cache_info().counters.miss, 10);
        assert_eq!(mmap_directory.get_cache_info().mmapped.len(), 10);

        for path in paths.iter() {
            let _r = mmap_directory.open_read(path).unwrap();
            assert_eq!(mmap_directory.get_cache_info().mmapped.len(), 10);
        }

        assert_eq!(mmap_directory.get_cache_info().counters.hit, 20);
        assert_eq!(mmap_directory.get_cache_info().counters.miss, 10);
        assert_eq!(mmap_directory.get_cache_info().mmapped.len(), 10);
        drop(keep);
        for path in paths.iter() {
            let _r = mmap_directory.open_read(path).unwrap();
            assert_eq!(mmap_directory.get_cache_info().mmapped.len(), 1);
        }
        assert_eq!(mmap_directory.get_cache_info().counters.hit, 20);
        assert_eq!(mmap_directory.get_cache_info().counters.miss, 20);
        assert_eq!(mmap_directory.get_cache_info().mmapped.len(), 0);

        for path in &paths {
            mmap_directory.delete(path).unwrap();
        }
        assert_eq!(mmap_directory.get_cache_info().counters.hit, 20);
        assert_eq!(mmap_directory.get_cache_info().counters.miss, 20);
        assert_eq!(mmap_directory.get_cache_info().mmapped.len(), 0);
        for path in paths.iter() {
            assert!(mmap_directory.open_read(path).is_err());
        }
        assert_eq!(mmap_directory.get_cache_info().counters.hit, 20);
        assert_eq!(mmap_directory.get_cache_info().counters.miss, 30);
        assert_eq!(mmap_directory.get_cache_info().mmapped.len(), 0);
    }

    fn assert_eventually<P: Fn() -> Option<String>>(predicate: P) {
        for _ in 0..30 {
            if predicate().is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        if let Some(error_msg) = predicate() {
            panic!("{}", error_msg);
        }
    }

    #[test]
    fn test_mmap_released() {
        let mmap_directory = MmapDirectory::create_from_tempdir().unwrap();
        let mut schema_builder: SchemaBuilder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();

        {
            let index =
                Index::create(mmap_directory.clone(), schema, IndexSettings::default()).unwrap();

            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
            let mut log_merge_policy = LogMergePolicy::default();
            log_merge_policy.set_min_num_segments(3);
            index_writer.set_merge_policy(Box::new(log_merge_policy));
            for _num_commits in 0..10 {
                for _ in 0..10 {
                    index_writer.add_document(doc!(text_field=>"abc")).unwrap();
                }
                index_writer.commit().unwrap();
            }

            let reader = index
                .reader_builder()
                .reload_policy(ReloadPolicy::Manual)
                .try_into()
                .unwrap();

            for _ in 0..4 {
                index_writer.add_document(doc!(text_field=>"abc")).unwrap();
                index_writer.commit().unwrap();
                reader.reload().unwrap();
            }
            index_writer.wait_merging_threads().unwrap();

            reader.reload().unwrap();
            let num_segments = reader.searcher().segment_readers().len();
            assert!(num_segments <= 4);
            let num_components_except_deletes_and_tempstore =
                crate::index::SegmentComponent::iterator().len() - 2;
            let max_num_mmapped = num_components_except_deletes_and_tempstore * num_segments;
            assert_eventually(|| {
                let num_mmapped = mmap_directory.get_cache_info().mmapped.len();
                if num_mmapped > max_num_mmapped {
                    Some(format!(
                        "Expected at most {max_num_mmapped} mmapped files, got {num_mmapped}"
                    ))
                } else {
                    None
                }
            });
        }
        // This test failed on CI. The last Mmap is dropped from the merging thread so there might
        // be a race condition indeed.
        assert_eventually(|| {
            let num_mmapped = mmap_directory.get_cache_info().mmapped.len();
            if num_mmapped > 0 {
                Some(format!("Expected no mmapped files, got {num_mmapped}"))
            } else {
                None
            }
        });
    }
}
