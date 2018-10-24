#! /bin/bash

THUNDER_URL=''
THREADS=5
IS_WGET=0
IS_MWGET=0
IS_AXEL=0

# Check if we have mwget / axel / wget
MWGET=$(which mwget)
AXEL=$(which axel)
if [ -n $MWGET ]; then
    IS_MWGET=1
elif [ -n $AXEL ]; then
    IS_AXEL=1
else
    IS_WGET=1
fi

# Usage print function
function usage {
    printf "
Usage: args [-h] [-n num_threads] [-m] [-a] -u thunder_url

  -h, Print this help
  -n NUM, Specify threads number for multi download, default 5
  -m, Use mwget for multi downloading, default one
  -w, use wget for downloading
  -a, use axel for multi downloading
  -u URL, Thunder url

"
    exit 0
}

# parse arguments
while getopts n:u:wmah option
do
  case "$option" in
    n)
      THREADS=$OPTARG;;
    u)
      THUNDER_URL=$OPTARG;;
    w)
      IS_WGET=1
      IS_MWGET=0
      IS_AXEL=0;;
    m)
      IS_WGET=0
      IS_MWGET=1
      IS_AXEL=0;;
    a)
      IS_WGET=0
      IS_MWGET=0
      IS_AXEL=1;;
    h)
      usage;;
  esac
done

# Check if url is provided as positional argument
shift $((OPTIND - 1))
# echo "Options not pared: $@"
if [ -n $1 ] && [ -z $THUNDER_URL ]; then
    THUNDER_URL=$1
fi

if [ -z $THUNDER_URL ]; then
    echo "Empty url"
    usage
fi

echo ""
echo "url = $THUNDER_URL, num threads = $THREADS, use wget = $IS_WGET, use mwget = $IS_MWGET, use axel = $IS_AXEL"

# Decode thunder url by base64
# thunder://{base64}[/] --> {base64}[/] --> base64 --> AA{real url}ZZ --> real url
THUNDER_URL=${THUNDER_URL[@]/"thunder://"/}
if [ ${THUNDER_URL:0-1} = '/' ]; then
    THUNDER_URL=${THUNDER_URL:0:0-1}
fi
REAL_URL=$(echo $THUNDER_URL | base64 -d)
if [ $? -ne 0 ]; then
    echo "Base64 decode error, base64: $THUNDER_URL"
    exit 1
fi
REAL_URL=${REAL_URL:2:0-2}

# Python decode
# REAL_URL=$(python3 -c '
# import base64
# THUNDER_HEADER = "thunder://"
# url = "'$THUNDER_URL'"
# if url.startswith(THUNDER_HEADER):
#     url = url[len(THUNDER_HEADER):]
#     if url.endswith("/"):
#         url = url[:-1]
#     print(base64.b64decode(url).decode()[2:-2])
# ')

echo "Real url = $REAL_URL"
echo ""

# Multi download
read -p "We go? (Y/n):" RESULT
if [[ ! $RESULT =~ ^[yY] ]]; then
    echo 'So we exit'
    exit 0
else
    if [ $IS_WGET -eq 1 ]; then
        echo 'Cool wget!'
        wget $REAL_URL
    elif [ $IS_AXEL -eq 1 ]; then
        echo 'Cool axel!'
        axel -n $THREADS $REAL_URL
    else
        echo 'Cool mwget!'
        mwget -n $THREADS $REAL_URL
    fi
fi
