#!/bin/bash
for i in {1..4345}
do
  filename="diagram${i}"
  pdflatex -escape-shell "data/diagram_files/${filename}.tex"
  convert -density 600x600 "${filename}.pdf" -quality 90 -background white -alpha remove -resize 540x400 "${filename}.png"
  if [ "$i" -le 4000 ]
  then
    mv "${filename}.png" "data/train_images/${filename}.png"
  elif [ "$i" -le 4175 ]
  then
    mv "${filename}.png" "data/dev_images/${filename}.png"
  else
    mv "${filename}.png" "data/test_images/${filename}.png"
  fi
  rm "${filename}.aux"
  rm "${filename}.log"
  rm "${filename}.pdf"
done
