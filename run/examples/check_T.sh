gpr=(graphdot sklearn)
kernel=(graph vector preCalc)
mode=(loocv train_test)
p=pvap-lg
for((i=0;i<${#gpr[@]};++i))
do
	for((j=0;j<${#mode[@]};++j))
	do
		for((k=0;k<${#kernel[@]};++k))
		do
			g=${gpr[$i]}
			if [ "$g" == "graphdot" ]; then
				opt="L-BFGS-B"
			elif [ "$g" == "sklearn" ]; then
				opt="fmin_l_bfgs_b"
			fi
			m=${mode[$j]}
			kn=${kernel[$k]}
			if [ "$kn" == "preCalc" ]; then
				python3 ../KernelCalc.py -i $p.txt --result_dir check-$g-$kn-$m	 
				python3 ../KernelCalc.py -i $p.txt --result_dir check-$g-$kn-$m-normal		--normalized
				python3 ../GPR.py --gpr $g --kernel $kn --add_features rel_T --hyper_features 0.1 -i $p.txt --property $p --mode $m --alpha 10.0 --result_dir check-$g-$kn-$m				--optimizer None							> check-$p-$g-$kn-$m.log
				python3 ../GPR.py --gpr $g --kernel $kn --add_features rel_T --hyper_features 0.1 -i $p.txt --property $p --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-normal		--optimizer None --normalized				> check-$p-$g-$kn-$m-normal.log
			elif [ "$kn" == "vector" ]; then
				python3 ../GPR.py --gpr $g --kernel $kn --add_features rel_T --hyper_features 0.1 -i $p.txt --property $p --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-opt			--optimizer $opt							> check-$p-$g-$kn-$m-opt.log
			else
				python3 ../GPR.py --gpr $g --kernel $kn --add_features rel_T --hyper_features 0.1 -i $p.txt --property $p --mode $m --alpha 10.0 --result_dir check-$g-$kn-$m				--optimizer None							> check-$p-$g-$kn-$m.log
				python3 ../GPR.py --gpr $g --kernel $kn --add_features rel_T --hyper_features 0.1 -i $p.txt --property $p --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-normal		--optimizer None --normalized				> check-$p-$g-$kn-$m-normal.log
				python3 ../GPR.py --gpr $g --kernel $kn --add_features rel_T --hyper_features 0.1 -i $p.txt --property $p --mode $m --alpha 10.0 --result_dir check-$g-$kn-$m-opt			--optimizer $opt							> check-$p-$g-$kn-$m-opt.log
				python3 ../GPR.py --gpr $g --kernel $kn --add_features rel_T --hyper_features 0.1 -i $p.txt --property $p --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-opt-normal	--optimizer $opt --normalized				> check-$p-$g-$kn-$m-opt-normal.log
				python3 ../GPR.py --gpr $g --kernel $kn --add_features rel_T --hyper_features 0.1 -i $p.txt --property $p --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-opt-normal	--optimizer $opt --normalized --load_model	> check-$p-$g-$kn-$m-opt-normal-load.log
			fi
		done
	done
done
mv ../check-* .
