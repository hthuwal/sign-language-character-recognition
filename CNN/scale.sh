for i in *
do
	if [ -d $i ]
	then
		cd $i
		for j in *
		do
			echo $j
			mogrify -scale 224x224\! $j
		done
		cd ..
	fi
done
