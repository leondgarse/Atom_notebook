# ___2017 - 01 - 14 Github___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2017 - 01 - 14 Github___](#2017-01-14-github)
  - [目录](#目录)
  - [github相关命令](#github相关命令)
  - [Setting up Git](#setting-up-git)
  - [](#)
  - [Adding a new SSH key to your GitHub account](#adding-a-new-ssh-key-to-your-github-account)
  - [Adding a new repository](#adding-a-new-repository)
  - [Create a local clone of your fork](#create-a-local-clone-of-your-fork)
  - [Configure Git to sync your fork with the original Spoon-Knife repository](#configure-git-to-sync-your-fork-with-the-original-spoon-knife-repository)
  - [Syncing a fork](#syncing-a-fork)

  <!-- /TOC -->
***

# github相关命令
  - Tell Git your name so your commits will be properly labeled:
    ```
    git config --global user.name "YOUR NAME"
    ```
    Tell Git the email address that will be associated with your Git commits.
    ```
    git config --global user.email "YOUR EMAIL ADDRESS"
    ```
    Get current configuration
    ```
    git config --get user.email
    git config --get user.name
    ```
    Set git to use the credential memory cache
    ```
    git config --global credential.helper cache
    ```
    Set the cache to timeout after 1 hour (setting is in seconds)
    ```
    git config --global credential.helper 'cache --timeout=3600'
    ```
  - create a new repository on the command line
    ```
    echo "# testgit" >> README.md
    git init
    git add README.md
    git commit -m "first commit"
    git remote add origin https://github.com/leondgarse/foo.git
    git push -u origin master
    ```
  - Create a local clone of your fork
    ```
    git clone https://github.com/YOUR-USERNAME/Spoon-Knife
    ```
    see the current configured remote repository for your fork.
    ```
    git remote -v
    ```
    Add an upstream repository
    ```
    git remote add upstream https://github.com/octocat/Spoon-Knife.git
    git remote -v
    ```
    Fetch the branches and their respective commits from the upstream repository. Commits to master will be stored in a local branch, upstream/master.
    ```
    git fetch upstream
    ```
    Check out your fork's local master branch.
    ```
    git checkout master
    ```
    Merge the changes from upstream/master into your local master branch. This brings your fork's master branch into sync with the upstream repository, without losing your local changes.
    ```
    git merge upstream/master
    ```
***

# Setting up Git
  ```python
  Download and install the latest version of Git.
  On your computer, open your command line application.
  Tell Git your name so your commits will be properly labeled. Type everything after the $ here:
          git config --global user.name "YOUR NAME"
  Tell Git the email address that will be associated with your Git commits. The email you specify should be the same one found in your email settings. To keep your email address hidden, see "Keeping your email address private".
          git config --global user.email "YOUR EMAIL ADDRESS"
  ```
***

#
  ```python
  If you're cloning GitHub repositories using HTTPS, you can use a credential helper to tell Git to remember your GitHub username and password every time it talks to GitHub.
  If you clone GitHub repositories using SSH, then you authenticate using SSH keys instead of a username and password. For help setting up an SSH connection, see Generating an SSH Key.
  Tip: You need Git 1.7.10 or newer to use the credential helper.
  Turn on the credential helper so that Git will save your password in memory for some time. By default, Git will cache your password for 15 minutes.
  In Terminal, enter the following:
  git config --global credential.helper cache
  # Set git to use the credential memory cache
  To change the default password cache timeout, enter the following:
  git config --global credential.helper 'cache --timeout=3600'
  # Set the cache to timeout after 1 hour (setting is in seconds)
  ```
***

# Adding a new SSH key to your GitHub account
  ```python
  Adding a new SSH key to your GitHub account
  To configure your GitHub account to use your new (or existing) SSH key, you'll also need to add it to your GitHub account.
  Before adding a new SSH key to your GitHub account, you should have:
  Checked for existing SSH keys
  Generated a new SSH key and added it to the ssh-agent
  Copy the SSH key to your clipboard.
  If your SSH key file has a different name than the example code, modify the filename to match your current setup. When copying your key, don't add any newlines or whitespace.
  $ sudo apt-get install xclip
  # Downloads and installs xclip. If you don't have `apt-get`, you might need to use another installer (like `yum`)

  $ xclip -sel clip < ~/.ssh/id_rsa.pub
  # Copies the contents of the id_rsa.pub file to your clipboard
  Tip: If xclip isn't working, you can locate the hidden .ssh folder, open the file in your favorite text editor, and copy it to your clipboard.
  In the upper-right corner of any page, click your profile photo, then click Settings.
  In the user settings sidebar, click SSH and GPG keys.
  Click New SSH key or Add SSH key.
  In the "Title" field, add a descriptive label for the new key. For example, if you're using a personal Mac, you might call this key "Personal MacBook Air".
  Paste your key into the "Key" field.
  Click Add SSH key.
  If prompted, confirm your GitHub password.
  ```
  - 测试连接：
    ```
    $ ssh -T git@github.com
    Warning: Permanently added the RSA host key for IP address '192.30.253.112' to the list of known hosts.
    Hi leondgarse! You've successfully authenticated, but GitHub does not provide shell access.
    ```
***

# Adding a new repository
  ```python
  Quick setup — if you’ve done this kind of thing before
  …or create a new repository on the command line
  echo "# testgit" >> README.md
  git init
  git add README.md
  git commit -m "first commit"
  git remote add origin https://github.com/leondgarse/foo.git
  git push -u origin master
  …or push an existing repository from the command line
  git remote add origin https://github.com/leondgarse/foo.git
  git push -u origin master
  …or import code from another repository
  You can initialize this repository with code from a Subversion, Mercurial, or TFS project.
  ```
***

# Create a local clone of your fork
  ```python
  Right now, you have a fork of the Spoon-Knife repository, but you don't have the files in that repository on your computer. Let's create a clone of your fork locally on your computer.
  On GitHub, navigate to your fork of the Spoon-Knife repository.
  In the right sidebar of your fork's repository page, click
  to copy the clone URL for your fork.
  Open Terminal (for Mac and Linux users) or the command prompt (for Windows users).
  Type git clone, and then paste the URL you copied in Step 2. It will look like this, with your GitHub username instead of YOUR-USERNAME:
  git clone https://github.com/YOUR-USERNAME/Spoon-Knife
  Press Enter. Your local clone will be created.
  ```
***

# Configure Git to sync your fork with the original Spoon-Knife repository
  ```python
  When you fork a project in order to propose changes to the original repository, you can configure Git to pull changes from the original, or upstream, repository into the local clone of your fork.
  On GitHub, navigate to the octocat/Spoon-Knife repository.
  In the right sidebar of the repository page, click
  to copy the clone URL for the repository.
  Open Terminal (for Mac and Linux users) or the command prompt (for Windows users).
  Change directories to the location of the fork you cloned in Step 2: Create a local clone of your fork.
  To go to your home directory, type just cd with no other text.
  To list the files and folders in your current directory, type ls.
  To go into one of your listed directories, type cd your_listed_directory.
  To go up one directory, type cd ...
  Type git remote -v and press Enter. You'll see the current configured remote repository for your fork.
  git remote -v
  origin https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
  origin https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
  Type git remote add upstream, and then paste the URL you copied in Step 2 and press Enter. It will look like this:
  git remote add upstream https://github.com/octocat/Spoon-Knife.git
  To verify the new upstream repository you've specified for your fork, type git remote -v again. You should see the URL for your fork as origin, and the URL for the original repository as upstream.
  git remote -v
  origin  https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
  origin  https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
  upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git (fetch)
  upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git (push)
  ```
***

# Syncing a fork
  ```python
  Sync a fork of a repository to keep it up-to-date with the upstream repository.
  Tip: Before you can sync your fork with an upstream repository, you must configure a remote that points to the upstream repository in Git.
  Open Terminal (for Mac and Linux users) or the command prompt (for Windows users).
  Change the current working directory to your local project.
  Fetch the branches and their respective commits from the upstream repository. Commits to master will be stored in a local branch, upstream/master.
  git fetch upstream
  remote: Counting objects: 75, done.
  remote: Compressing objects: 100% (53/53), done.
  remote: Total 62 (delta 27), reused 44 (delta 9)
  Unpacking objects: 100% (62/62), done.
  From https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY
   * [new branch]   master   -> upstream/master
  Check out your fork's local master branch.
  git checkout master
  Switched to branch 'master'
  Merge the changes from upstream/master into your local master branch. This brings your fork's master branch into sync with the upstream repository, without losing your local changes.
  git merge upstream/master
  Updating a422352..5fdff0f
  Fast-forward
   README          |  9 -------
   README.md         |  7 ++++++
   2 files changed, 7 insertions(+), 9 deletions(-)
   delete mode 100644 README
   create mode 100644 README.md
  If your local branch didn't have any unique commits, Git will instead perform a "fast-forward":
  git merge upstream/master
  Updating 34e91da..16c56ad
  Fast-forward
   README.md         |  5 +++--
   1 file changed, 3 insertions(+), 2 deletions(-)
  ```
  ```sh
  # Add the remote, call it "upstream":
  git remote add upstream https://github.com/whoever/whatever.git

  # Fetch all the branches of that remote into remote-tracking branches
  git fetch upstream

  # Make sure that you're on your master branch:
  git checkout master

  # Rewrite your master branch so that any commits of yours that
  # aren't already in upstream/master are replayed on top of that
  # other branch:
  git rebase upstream/master

  # git push -f origin master
  git push
  ```
***

# Clean all github history
  - [Steps to clear out the history of a git/github repository](https://gist.github.com/stephenhardy/5470814)
  ```sh
  REMOTE_BACKUP=`git remote -v | awk 'NR==1 {print $2}'`
  echo $REMOTE_BACKUP

  # Remove the history from
  rm -rf .git

  # recreate the repos from the current content only
  git init
  git add .
  git commit -m "Clear out the history"

  # push to the github remote repos ensuring you overwrite history
  git remote add origin $REMOTE_BACKUP
  git push -u --force origin master
  ```
***
