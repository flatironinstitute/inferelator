REPOSRC=https://github.com/simonsfoundation/kvsstcp
LOCALREPO=$(pwd)/kvsstcp
LOCALREPO_VC_DIR=$LOCALREPO/.git

if [ ! -d $LOCALREPO_VC_DIR ]
then
    git clone $REPOSRC $LOCALREPO
else
    pushd $LOCALREPO
    git pull $REPOSRC
    popd
fi

export PYTHONPATH=$PYTHONPATH:$LOCALREPO

python $LOCALREPO/kvsstcp.py --execcmd "nosetests -v"
