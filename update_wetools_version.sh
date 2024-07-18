#!/bin/bash
POSITIONAL_ARGS=()

# Get current version:
VERSION=$(grep -w "version =" pyproject.toml )
VERSION=${VERSION:11:10}
VERSION=${VERSION%?}
echo 'Current version:' $VERSION

# Command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--version)
      # commit message
      NEW_VERSION="$2"
      shift
      shift
      ;;
    -m|--message)
      # commit message
      MESSAGE="$2"
      shift
      shift
      ;;
  esac
done


# Update version if no custom version set
if [[ "$NEW_VERSION" == "" ]]
then
  # Create new version:
  pre=${VERSION%.*}
  # Get last subversion and add one:
  aft=${VERSION#*.}
  aft=${aft#*.}
  NEW_VERSION=$pre"."$(($aft +1))
fi
echo "New version:    " $NEW_VERSION

# Make default message:
if [[ $MESSAGE == "" ]]
then
  MESSAGE="Version "$NEW_VERSION
fi
echo "Message:        " $MESSAGE


# Update the version in pyproject toml
value=`cat pyproject.toml`

# Get stuff before/after the version
v1=${value%$VERSION*}
v2=${value#*$VERSION}

# Change stuff in between and output to toml
v1+="$NEW_VERSION"
v1+=$v2
echo "$v1" > pyproject.toml

# Add to the commit
git add pyproject.toml

git commit -m $MESSAGE
git tag -a -m $MESSAGE $NEW_VERSION