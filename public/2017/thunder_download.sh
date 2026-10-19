#! /bin/bash

THUNDER_URL=''
THREADS=5

# Check if we have mwget / axel / wget
DOWNLOADER="wget"
if [ -n "$(which mwget)" ]; then
    DOWNLOADER="mwget"
elif [ -n "$(which axel)" ]; then
    DOWNLOADER="axel"
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
  -d DOWNLOADER, specify other downloader like aria2c

"
    exit 0
}

# parse arguments
while getopts n:u:d:wmah option
do
  case "$option" in
    n)
      THREADS=$OPTARG;;
    u)
      THUNDER_URL=$OPTARG;;
    w)
      DOWNLOADER="wget";;
    m)
      DOWNLOADER="mwget";;
    a)
      DOWNLOADER="axel";;
    d)
      DOWNLOADER=$OPTARG;;
    h)
      usage;;
  esac
done

# Check if url is provided as positional argument
shift $((OPTIND - 1))
# echo "Options not pared: $@"
if [ -n "$1" ] && [ -z "$THUNDER_URL" ]; then
    THUNDER_URL=$1
fi

if [ -z "$THUNDER_URL" ]; then
    echo "Empty url"
    usage
fi

if [ $DOWNLOADER = "mwget" ] || [ $DOWNLOADER = "axel" ]; then
    DOWNLOADER="$DOWNLOADER -n $THREADS"
fi

echo ""
echo "url = $THUNDER_URL, Downloader = $DOWNLOADER"

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
    echo "$DOWNLOADER $REAL_URL"
    $DOWNLOADER $REAL_URL
fi
