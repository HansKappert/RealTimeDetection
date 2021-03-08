du -a|tr '[:upper:]' '[:lower:]' |grep '.jp' | sed '/.*\.\/.*\/.*/!d' | cut -d/ -f2 | sort | uniq -c 
