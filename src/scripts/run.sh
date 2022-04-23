input_folder='/home/olga/deblurnet/experiments/Unet'
output_folder='/home/olga/deblurnet/results/experiments/Unet'
app_folder='/home/olga/deblurnet'

# install crudini package if needed
if apt-get -qq install crudini; then
	echo "Required crudini package found"
	echo
else
	echo "Installing crudini"
	apt-get install crudini
	echo
fi

# get a list of all files in input folder
echo "Collecting configuration files for the experiment from: $input_folder"
input_configs=()
for f in "$input_folder"/*
do
	input_configs+=($f)
done

# create output folder if doesn't exist
echo "Creating output folder if doesn't exist: $output_folder"
mkdir -p $output_folder

# activate python env
source "$app_folder/venv/bin/activate"

for c in ${input_configs[@]}
do
	# get configuration name
	name=$(basename "$c" ".ini")
	echo
	echo "---------- Processing configuration: $name"
	mkdir -p "$output_folder/$name"

	# copy config file
	cp $c "$output_folder/$name/config.ini"

	# update settings in current config file
	crudini --set "$output_folder/$name/config.ini" general log_path "$output_folder/$name/log.txt"
	crudini --set "$output_folder/$name/config.ini" general log_to_file True
	model_path=`crudini --get "$output_folder/$name/config.ini" results model_path`
	prediction_path=`crudini --get "$output_folder/$name/config.ini" results prediction_path`
	crudini --set "$output_folder/$name/config.ini" results model_path "$output_folder/$name/${model_path:2}"
	crudini --set "$output_folder/$name/config.ini" results prediction_path "$output_folder/$name/${prediction_path:2}"

	# run deblurnet
	python "$app_folder/deblurnet.py" "$output_folder/$name/config.ini"

	# collect results
	if [ $? -eq 0 ]; then
	    	echo "---------- Experiment $name COMPLETED successfully"
	else
	    	echo "---------- Experiment $name FAILED"
	fi
done

# update rights
chmod -R a+w "$output_folder"

