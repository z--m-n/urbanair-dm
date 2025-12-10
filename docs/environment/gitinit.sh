#!/bin/bash

locs=(
"analysis/" 
"metadata/"
)

sublocs=(
"conf" 
"data" 
"notebooks"
)

function gitignore {
	local gp=$1
	shift
	local ga=("$@")
		
	local gi="${gp}/.gitignore"
	echo "${gi}"
	echo -e "# (urbanair-dm) Custom git exclude paths" >> ${gi}
	echo -e "" >> ${gi}
	echo -e "## Private assets and runtime outputs" >> ${gi}
	echo -e "_tmp_*" >> ${gi}
	echo -e "_run_*" >> ${gi}
	for a in "${ga[@]}"; do
		echo -e "${a}" >> ${gi}
	done;
}

for n in "${locs[@]}" ; do
    echo "$n"
	mkdir -p "${n}";
	ga=( "conf/secrets.toml", "data/", "logs/", "tmp/" )
	gitignore "${n}" "${ga[@]}"
	
	for m in "${sublocs[@]}"; do
		mkdir -p "${n}/${m}";
		ga=( "" )
		gitignore "${n}/${m}" "${ga[@]}"
	done;
	
	md="./${n}/README.md"	
	touch "${md}"
	echo -e "# (urbanair-dm) ${n}" >> ${md}		
done;

