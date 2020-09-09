

url="http://duckduckgo.com/?q=\\"
# comp=$1
#url+=comp
url="${url} ${1}"

w3m "$url" -dump
w3m "$url" -dump>tmp.txt