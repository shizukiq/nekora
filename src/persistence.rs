use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const MAX_NOTE_BYTES: u64 = 1 << 20;
const MAX_RUNTIME_FILE_BYTES: u64 = 16 << 20;

pub fn ensure_directory(directory: &Path) -> bool {
    fs::create_dir_all(directory).is_ok() && directory.is_dir()
}

pub fn read_file(path: &Path) -> Option<String> {
    read_file_up_to(path, MAX_NOTE_BYTES)
}

pub fn read_runtime_file(path: &Path) -> Option<String> {
    read_file_up_to(path, MAX_RUNTIME_FILE_BYTES)
}

fn read_file_up_to(path: &Path, max_bytes: u64) -> Option<String> {
    let size = fs::metadata(path).ok()?.len();
    if size > max_bytes {
        return None;
    }
    fs::read_to_string(path).ok()
}

pub fn write_file_atomic(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(directory) = path
        .parent()
        .filter(|directory| !directory.as_os_str().is_empty())
    {
        fs::create_dir_all(directory)?;
    }
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
