//! The vault on disk: the few filesystem primitives the diary is built on.
//!
//! Nothing here is Nekora-specific; it is the small, careful layer that keeps a
//! note from being half-written. Writes go through a temp file and a rename so a
//! crash mid-write never leaves a truncated note the parser would then reject.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// A single note is prose plus one embedding line; a megabyte is far more than
// that will ever be, and the ceiling stops a corrupt or hostile file from being
// slurped whole into memory.
const MAX_NOTE_BYTES: u64 = 1 << 20;

pub fn ensure_directory(directory: &Path) -> bool {
    fs::create_dir_all(directory).is_ok() && directory.is_dir()
}

/// Read a note, refusing anything larger than a note has any business being.
/// Returns `None` for a missing, oversized, or unreadable file so the caller can
/// simply skip it rather than abort loading the whole vault.
pub fn read_file(path: &Path) -> Option<String> {
    let size = fs::metadata(path).ok()?.len();
    if size > MAX_NOTE_BYTES {
        return None;
    }
    fs::read_to_string(path).ok()
}

/// Write `contents` to `path` atomically: fill a sibling `.tmp` and rename it
/// over the target, so a reader never sees a partial note. The rename is atomic
/// within a filesystem, which the vault always is.
pub fn write_file_atomic(path: &Path, contents: &str) -> io::Result<()> {
    let temporary = path.with_extension("md.tmp");
    {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

/// Every `.md` note in the vault, unsorted. The diary sorts them itself so the
/// id order is its concern, not the directory walk's.
pub fn markdown_files(directory: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "md") {
            files.push(path);
        }
    }
    Ok(files)
}
