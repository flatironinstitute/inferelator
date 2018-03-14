# Development notes

To allow developer freedom coupled with shared code safety
we ask developers on this project to develop in their own
*forked* copy of the project.  Changes intended to be added
to the primary project should always be offered as *pull requests*
to the project.  The pull requests should be examined and discussed
by the community before being merged into the project by another
developer who did not propose the pull request.

This project will loosely follow the development protocol
outlined in the [Numpy/scipy dev documentation](http://docs.scipy.org/doc/numpy/dev/)

## Tools

To do development you will need your own computer with the following installed

- The `git` change control tool suite.
- A good code editor like visual studio `code` or `sublime`.
- Python with related libraries (it is best to override any system Python).
- The R command line tool.

You will also need

- A `github` login set up to authenticate from your computer.
- Your own github fork of the repository.
- A clone of your fork of the repository in your computer.

# Ubuntu Instructions

These are the commands that I (Kostya) ran on Ubuntu:

## Fork the simonsfoundation repository (once)

In order to make changes to the repository you will work in your own copy
where you have complete freedom to try anything you like.  To get this copy
set up you need to create a "fork" of the primary repository and set up the
primary repository as a "remote".

To create your fork log in to github and go to 
[https://github.com/simonsfoundation/inferelator_ng](https://github.com/simonsfoundation/inferelator_ng)
and click "fork"

## Install the necessary packages (once)

In order to work with the forked repository you will need `git` and a number of
other tools.

```
sudo apt-get install python-dev
sudo apt-get install python-pip
pip install pandas
sudo apt-get install python-nose
sudo apt-get install git
sudo apt-get -y install r-base
```

## Configuring your `git` command line interface

You might want to follow these instructions in case your name and email are not set properly (you should only need to do this once):

```
    git config --global user.name "Your Name"
    git config --global user.email you@example.com
```
This allows git to associate your name and email with any changes you make.

## Cloning the forked `inferelator_ng` directory onto your machine

Go to the fork of `inferelator_ng` that's in your github (of form https://github.com/$USERNAME/inferelator_ng
also linked from your github profile page), and click "clone or download" to get `$URL`

To clone the fork onto your workstation type in the terminal:

```
git clone $URL
```

For me, this was:

```git clone https://github.com/kostyat/inferelator_ng.git```

## Set up the "remote" repository (once)

You will want to periodically 
[merge changes from the main repository into your fork](https://help.github.com/articles/syncing-a-fork).
This does not happen automatically -- to make it happen you need to 
[set up the main repository as a "remote"](https://help.github.com/articles/configuring-a-remote-for-a-fork)

`git remote add upstream https://github.com/simonsfoundation/inferelator_ng.git`

## Merging changes from the remote into your fork (as needed)

To merge changes from the main remote repository into your fork 
[github suggests](https://help.github.com/articles/syncing-a-fork) the following
sequence

```
# Fetch the upstream changes
git fetch upstream
# Make sure you are in the master branch of your fork
git checkout master
# Merge the upstream master to your master branch.
git merge upstream/master
# Push the merge master back to your fork at github.
git push
```

It is a good idea to merge the latest changes before you start any development
which you want to contribute back to the project to help avoid possible conflicts.

## Running the unit tests

Unit tests attempt to check that everything is working properly.
It is a good idea to run unit tests frequently, especially before making
changes and after making changes but before committing them.

## Running the unit tests with parallelism support from KVS

You can run the tests including the ones that require KVS using the
following command line:

```bash run_unittests.sh```

The resulting output will include print statements from the tests and also log
statements from the `kvsstcp` manager process, testing the interaction with KVS.

An alternative way to run tests is to switch into the inferelator_ng directory:

```cd inferelator_ng/```

run 
[nosetests](http://pythontesting.net/framework/nose/nose-introduction/) 
(this runs the unit tests):

```bash
export SKIP_KVS_TESTS=true
nosetests
```

The `SKIP_KVS_TESTS` environment variable suppresses tests that require
external parallelism support from the 
[KVS package](https://github.com/flatironinstitute/kvsstcp).

Output should look like this:

```
.......S...............
----------------------------------------------------------------------
Ran 23 tests in 2.179s

OK (SKIP=1)
```

Each dot stands for a unit test that ran, "S" stands for "Skipped".  If there are
failures the output will be more extensive, describing which tests failed and how.

For debugging purposes it is sometimes useful to use `print` statements and invoke
nosetests with the `--nocapture` option in order to see the output.

```bash
export SKIP_KVS_TESTS=true
nosetests --nocapture
```

# Making a contribution to the project

Before you make a change you want to contribute to the project it
is a good idea to make sure you have the latest code by merging
changes from the remote repository into your fork as described above.

To add a contribution you must change the repository, commit
your changes, push the changes to your fork, and make a pull request.
When your pull request is approved (by someone else) your contribution is complete.

To test the process you can go try the following:

```
cd inferelator_ng/
```

Change one of the files (for example the `utils.py` file), by adding a blank line or something.

## Run Unit Tests again to make sure everything still works

run nosetests (this runs the unit tests):

```
nosetests
```

## Push your changes to your Github directory

For each file you altered, run the following command:

```
git add utils.py
```

After you've done this for every file you've changed (in this case it's just 1 file), commit the changes to your fork by running:

```
git commit -m "fixed utils.py"
```
It is a good idea to commit the files you intended to change one at a time
to make sure you don't add unintended changes to the commit.

Finally push the changes to your fork:
```
git push
```

## Create a pull request

After you've done that, you should be able to find your committed modifications on your github fork web interface.

To request to add these changes to the master repository, go to your github and click "new pull request".
The target for the pull request should be the master branch of the main remote repository.
Someone with write access to the master repository will look over your changes.  They can comment,
approve, or close your request.

An approver may ask for changes before approving your pull request.  You can add changes by pushing
more commits (to the same branch of your forked repository, in this case the `master` branch).

## More sophisticated work flows

The workflow described above does not allow you to submit more than one pull request at a
time, for example -- to submit several pull requests 
at the same time you would need to create multiple branches
in your forked repository.  For more information on how to 
work with branches or do other more sophisticated interactions
please see the github documentation
pages and the scipy development documentation.
