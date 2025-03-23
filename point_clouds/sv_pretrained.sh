#!/bin/bash

# Function to download large files from Google Drive using gdown
download_gdrive() {
  # $1 = file ID
  # $2 = destination file name

  # Check if gdown is installed; if not, install it
  if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install --user gdown
  fi

  echo "Downloading $2 from Google Drive..."
  gdown --id "$1" -O "$2"
}

mkdir -p tmp
key="$1"

case $key in
  pretrained)
    download_gdrive 1SYwkAbahftSEm3ykHK-SdBrDi7hwa2BC tmp/pretrained.zip
    unzip -o tmp/pretrained.zip || { echo "Failed to unzip. Check if the download was successful."; exit 1; }
    ;;
  modelnet40)
    wget --no-check-certificate https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
    unzip modelnet40_ply_hdf5_2048.zip
    mv modelnet40_ply_hdf5_2048 data
    rm -r modelnet40_ply_hdf5_2048.zip
    download_gdrive 14Xcr8kG_1VFMpxpklH96U3d78k7lTuCq tmp/modelnet40_ply_hdf5_2048_valid_small.zip
    unzip -o tmp/modelnet40_ply_hdf5_2048_valid_small.zip
    mv modelnet40_ply_hdf5_2048_valid_small/* data/modelnet40_ply_hdf5_2048/
    rm -r modelnet40_ply_hdf5_2048_valid_small
    ;;
  *)
    echo "Unknown argument $1"
    ;;
esac

rm -r tmp
