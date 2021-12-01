#!/usr/bin/env bash

function get_checksum() {
   FILE=$1

   if [ -x "$(command -v md5sum)" ]; then
      # Linux
      MD5_RESULT=`md5sum ${FILE} | awk '{ print $1 }'`
   else
      # OS X
      MD5_RESULT=`md5 -q ${FILE}`
   fi
}


function download_file() {
   FILE=$1;
   CHECKSUM=$2;
   URL=$3;

   echo "downloading data ..."
   mkdir -p data
   cd data
   
   # Check if file already exists
   if [ -f ${FILE} ]; then
       echo "$FILE already exists, skipping download."
   else
      # Does not exists -> download
       wget -O - ${URL} > ${FILE}
       get_checksum ${FILE}
       if [ "${MD5_RESULT}" != "${CHECKSUM}" ]; then
           echo "error checksum does not match: run download again"
           exit -1
       else
           echo ${FILE} "checksum ok"
       fi
   fi
   
   cd ..
   echo "done"
}

# Format: download_file <file_name> <md5_checksum> <url>
download_file wiki_ts_200M_uint64 4f1402b1c476d67f77d2da4955432f7d https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/SVN8PI
download_file osm_cellids_800M_uint64 70670bf41196b9591e07d0128a281b9a https://www.dropbox.com/s/j1d4ufn4fyb4po2/osm_cellids_800M_uint64.zst?dl=1

