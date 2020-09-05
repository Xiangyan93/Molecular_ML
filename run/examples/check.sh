gpr=(graphdot sklearn)
kernel=(graph vector preCalc)
mode=(loocv train_test)
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
				python3 ../KernelCalc.py -i tt.txt --result_dir check-$g-$kn-$m	 
				python3 ../KernelCalc.py -i tt.txt --result_dir check-$g-$kn-$m-normal		--normalized
				python3 ../GPR.py --gpr $g --kernel $kn -i tt.txt --property tt --mode $m --alpha 1.0 --result_dir check-$g-$kn-$m				--optimizer None				> check-$g-$kn-$m.log
				python3 ../GPR.py --gpr $g --kernel $kn -i tt.txt --property tt --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-normal		--optimizer None --normalized	> check-$g-$kn-$m-normal.log
				if [ "$m" == "loocv" ]; then
					python3 ../GPR_active.py --gpr $g --kernel $kn -i tt.txt --property tt --optimizer None --alpha 0.01 result_dir check-$g-$kn-$m-normal --normalized --train_ratio 1.0 --learning_mode supervised 
			elif [ "$kn" == "vector" ]; then
				python3 ../GPR.py --gpr $g --kernel $kn -i tt.txt --property tt --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-opt			--optimizer $opt				> check-$g-$kn-$m-opt.log
			else
				python3 ../GPR.py --gpr $g --kernel $kn -i tt.txt --property tt --mode $m --alpha 1.0 --result_dir check-$g-$kn-$m				--optimizer None				> check-$g-$kn-$m.log
				python3 ../GPR.py --gpr $g --kernel $kn -i tt.txt --property tt --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-normal		--optimizer None --normalized	> check-$g-$kn-$m-normal.log
				python3 ../GPR.py --gpr $g --kernel $kn -i tt.txt --property tt --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-normal		--optimizer None --normalized --load_model > check-$g-$kn-$m-normal-load.log
				if [ "$g" == "graphdot" ]; then
					python3 ../GPR.py --gpr $g --kernel $kn -i tt.txt --property tt --mode $m --alpha 1.0 --result_dir check-$g-$kn-$m-opt			--optimizer $opt				> check-$g-$kn-$m-opt.log
					python3 ../GPR.py --gpr $g --kernel $kn -i tt.txt --property tt --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-opt-normal	--optimizer $opt --normalized	> check-$g-$kn-$m-opt-normal.log
					python3 ../GPR.py --gpr $g --kernel $kn -i tt.txt --property tt --mode $m --alpha 0.01 --result_dir check-$g-$kn-$m-opt-normal	--optimizer $opt --normalized --load_model	> check-$g-$kn-$m-opt-normal-load.log
				fi
			fi
		done
	done
done
mv ../check-* .
