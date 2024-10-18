//! File/code adapted from https://github.com/jamjamjon/usls
//!
//! Represents various directories on the system, including Home, Cache, Config, and current directory.
#[derive(Debug)]
pub enum FsAccess {
    Home,
    Cache,
    Config,
    Current,
}

#[allow(dead_code)]
impl FsAccess {
    pub fn save_out(subs: &[&str]) -> anyhow::Result<std::path::PathBuf> {
        Self::Current.raw_path_with_subs(subs)
    }

    /// Retrieves the base path for the specified directory type, optionally appending the `bvr` subdirectory.
    ///
    /// # Arguments
    /// * `raw` - If `true`, returns the base path without adding the `bvr` subdirectory.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The base path for the directory.
    fn get_path(&self, raw: bool) -> anyhow::Result<std::path::PathBuf> {
        let base_path = match self {
            FsAccess::Home => dirs::home_dir(),
            FsAccess::Cache => dirs::cache_dir(),
            FsAccess::Config => dirs::config_dir(),
            FsAccess::Current => std::env::current_dir().ok(),
        };

        let mut path = base_path.ok_or_else(|| {
            anyhow::anyhow!("Unsupported operating system. Supported OS: Linux, MacOS, Windows.")
        })?;

        if !raw {
            if let FsAccess::Home = self {
                path.push(".bvr");
            } else {
                path.push("bvr");
            }
        }
        Ok(path)
    }

    /// Returns the default path for the `bvr` directory, creating it automatically if it does not exist.
    ///
    /// Examples:
    /// `~/.cache/bvr`, `~/.config/bvr`, `~/.bvr`.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The default `bvr` directory path.
    pub fn path(&self) -> anyhow::Result<std::path::PathBuf> {
        let d = self.get_path(false)?;
        self.create_directory(&d)?;
        Ok(d)
    }

    /// Returns the raw path for the directory without adding the `bvr` subdirectory.
    ///
    /// Examples:
    /// `~/.cache`, `~/.config`, `~`.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The raw directory path.
    pub fn raw_path(&self) -> anyhow::Result<std::path::PathBuf> {
        self.get_path(true)
    }

    /// Constructs a path to the `bvr` directory with the provided subdirectories, creating it automatically.
    ///
    /// Examples:
    /// `~/.cache/bvr/sub1/sub2/sub3`, `~/.config/bvr/sub1/sub2`, `~/.bvr/sub1/sub2`.
    ///
    /// # Arguments
    /// * `subs` - A slice of strings representing subdirectories to append.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The resulting directory path.
    pub fn path_with_subs(&self, subs: &[&str]) -> anyhow::Result<std::path::PathBuf> {
        let mut d = self.get_path(false)?;
        self.append_subs(&mut d, subs)?;
        Ok(d)
    }

    /// Constructs a path to a specified directory with the provided subdirectories, creating it automatically.
    ///
    /// Examples:
    /// `~/.cache/sub1/sub2/sub3`, `~/.config/sub1/sub2`, `~/sub1/sub2`.
    ///
    /// # Arguments
    /// * `subs` - A slice of strings representing subdirectories to append.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The resulting directory path.
    pub fn raw_path_with_subs(&self, subs: &[&str]) -> anyhow::Result<std::path::PathBuf> {
        let mut d = self.get_path(true)?;
        self.append_subs(&mut d, subs)?;
        Ok(d)
    }

    /// Appends subdirectories to the given base path and creates the directories if they don't exist.
    fn append_subs(&self, path: &mut std::path::PathBuf, subs: &[&str]) -> anyhow::Result<()> {
        for sub in subs {
            path.push(sub);
        }
        self.create_directory(path)?;
        Ok(())
    }

    /// Creates the specified directory if it does not exist.
    fn create_directory(&self, path: &std::path::PathBuf) -> anyhow::Result<()> {
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        Ok(())
    }
}
