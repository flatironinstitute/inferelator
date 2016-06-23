#1. Fork
Go to https://github.com/simonsfoundation/inferelator_ng and click "fork"

#2. Install the necessary packages:

```
sudo apt-get install python-dev
sudo apt-get install python-pip
pip install pandas
sudo apt-get install python-nose
sudo apt-get install git
```

#3. Cloning the forked `inferelator_ng` directory onto your machine

Go to the fork of `inferelator_ng` that's in your github (https://github.com/$USERNAME/inferelator_ng), and click "clone or download" to get `$URL`

Type in the terminal:

```
git clone $URL
```

For me, this was:

```git clone https://github.com/kostyat/inferelator_ng.git```

#4. Running the unit tests

Switch into the inferelator_ng directory:

```cd inferelator_ng/```

run nosetests (this runs the unit tests):

```
nosetests
```

Output should look like this:

```
.......S...............
----------------------------------------------------------------------
Ran 23 tests in 2.179s

OK (SKIP=1)
```

Each dot stands for a unit test that ran, "S" stands for "Skipped".

#5. Modify your files

```
cd inferelator_ng/
```

Change one of the files (for example the `utils.py` file), by adding a blank line or something.

#6. Run Unit Tests again to make sure everything still works

run nosetests (this runs the unit tests):

```
nosetests
```

#7. Push your changes to your Github directory

For each file you altered, run the following command:

```
git add utils.py
```

After you've done this for every file you've changed (in this case it's just 1 file), commit the changes to your fork by running:

```
git commit -m "fixed utils.py"
```

You might want to follow these instructions in case your name and email are not set properly (you should only need to do this once):

```
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly:

    git config --global user.name "Your Name"
    git config --global user.email you@example.com

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author
```
Finally push the changes to your branch:
```
git push
```

#7. Pull request

After you've done that, you should be able to find your modifications on your github

Then, to add these changes to the master repository, go to your github and click "new pull request".

Someone with write access to the master directory will look over your changes.

#8. Keeping up with the master directory

If you made a change to your files on your Github page and you want to update the files on your machine to have those changes, run

`git pull`

Other people could make changes to the master directory. In order for your branch to get those changes, first run this command to add the upstream directory to the list of "remotes" to update from:

`git remote add upstream https://github.com/simonsfoundation/inferelator_ng.git`

And then pull the code from the master Github directory onto your machine:

`git pull upstream master`


