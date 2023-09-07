#!/usr/bin/env sh
JSON_CONVERTER_PY=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/json_converter.py

for f in $(find $1 -name '*[!.v2].json' -type f); do
    new_f="${f%.json}.json"
    echo "Converting $f -> $new_f"
    python $JSON_CONVERTER_PY $f $new_f
    if [ $? -eq 1 ]; then
        exit 1
    fi
done