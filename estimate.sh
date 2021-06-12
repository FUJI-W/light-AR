#!/bin/bash

help() {
  echo "Usage:"
  echo "    run.sh [-i INPUT_IMAGE] [-o OUTPUT_PATH] [-m MODE]"
  echo ""
  echo "Description:"
  echo "    INPUT_IMAGE, the path to input image"
  echo "    OUTPUT_PATH, the path to store results"
  echo "    MODE, the level of estimate net (0, 1, 2)"
  echo ""
}

while getopts "m:i:o:h:w:" OPT; do
  case $OPT in
    m) MODE="$OPTARG";;
    i) INPUT_IMAGE="$OPTARG";;
    o) OUTPUT_PATH="$OPTARG";;
    h) HEIGHT="$OPTARG";;
    w) WIDTH="$OPTARG";;
    ?) help ;;
  esac
done

#if [ $# != 3 ]; then
#  echo "Wrong Input! Try use -h to see the help."
#  exit 1
#fi

if [ -z "$MODE" ]; then
  MODE="1"
fi

if [ -z "$INPUT_IMAGE" ]; then
  INPUT_IMAGE="inputs/im.png"
fi

if [ -z "$OUTPUT_PATH" ]; then
  OUTPUT_PATH="outputs"
fi

if [ -z "$HEIGHT" ]; then
  HEIGHT="480"
fi

if [ -z "$WIDTH" ]; then
  WIDTH="640"
fi

echo "$MODE"
echo "$INPUT_IMAGE"
echo "$OUTPUT_PATH"

if [ "$MODE" = "0" ]; then
  python ./nets/test.py --cuda \
  --mode $MODE \
  --imPath "$INPUT_IMAGE" \
  --outPath "$OUTPUT_PATH" \
  --isLight \
  --level 1  \
  --experiment0 ./nets/models/check_cascade0_w320_h240 --nepoch0 14  \
  --experimentLight0 ./nets/models/check_cascadeLight0_sg12_offset1 --nepochLight0 10  \
  --imHeight0 $HEIGHT --imWidth0 $WIDTH
elif [ "$MODE" = "1" ]; then
  python ./nets/test.py --cuda \
  --mode $MODE \
  --imPath "$INPUT_IMAGE" \
  --outPath "$OUTPUT_PATH" \
  --isLight \
  --level 2  \
  --experiment0 ./nets/models/check_cascade0_w320_h240 --nepoch0 14  \
  --experimentLight0 ./nets/models/check_cascadeLight0_sg12_offset1 --nepochLight0 10  \
  --experiment1 ./nets/models/check_cascade1_w320_h240 --nepoch1 7  \
  --experimentLight1 ./nets/models/check_cascadeLight1_sg12_offset1 --nepochLight1 10  \
  --imHeight0 $HEIGHT --imWidth0 $WIDTH --imHeight1 $HEIGHT --imWidth1 $WIDTH
else
  echo "22222222"
  python ./nets/test.py --cuda \
  --mode $MODE \
  --imPath "$INPUT_IMAGE" \
  --outPath "$OUTPUT_PATH" \
  --isLight --isBS  \
  --level 2  \
  --experiment0 ./nets/models/check_cascade0_w320_h240 --nepoch0 14  \
  --experimentLight0 ./nets/models/check_cascadeLight0_sg12_offset1 --nepochLight0 10  \
  --experimentBS0 ./nets/models/checkBs_cascade0_w320_h240  \
  --experiment1 ./nets/models/check_cascade1_w320_h240 --nepoch1 7  \
  --experimentLight1 ./nets/models/check_cascadeLight1_sg12_offset1 --nepochLight1 10  \
  --experimentBS1 ./nets/models/checkBs_cascade1_w320_h240  \
  --imHeight0 $HEIGHT --imWidth0 $WIDTH --imHeight1 $HEIGHT --imWidth1 $WIDTH
fi

