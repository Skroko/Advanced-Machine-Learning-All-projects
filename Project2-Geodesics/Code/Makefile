#partA experiments

train_10_models_without_ensembles:
	seeds=(123 234 345 456 567 678 789 890 901 12); \
	for i in {1..10}; do \
		echo "Running iteration $$i"; \
		python3 main.py train --model "model$$i" --num-ensembles 1 --epochs 20 --seed $${seeds[$$i-1]}; \
	done

plot_10_models_without_ensembles:
	seeds=(123 234 345 456 567 678 789 890 901 12); \
	for i in {1..10}; do \
		echo "Running iteration $$i"; \
		python3 main.py plot_non_ensemble_model --model "model$$i" --num-ensembles 1 --epochs 20 --seed $${seeds[$$i-1]}; \
	done

#partB experiments
train_model_with_ensembles:
	python3 main.py train --model "ensemble_model" --num-ensembles 10 --epochs 20 --seed 123; \

plot_ensemble_model:
	python3 main.py plot_ensemble_model --model "ensemble_model" --num-ensembles 10 --epochs 20 --seed 123 --num-curves 50; \



# train_10_models_without_ensembles:
# 	#make list of seeds	
# 	seeds=(123 234 345 456 567 678 789 890 901 12); \
# 	for i in {1..10}; do \
# 		echo "Running iteration $$i"; \
# 		python3 partA.py train --model "model$$i" --num_ensembles 1 --epochs 100 --seed $${seeds[$$i-1]}; \
# 	done

# train_10_models_with_ensembles:
# 	#make list of seeds	
# 	seeds=(123 234 345 456 567 678 789 890 901 12); \
# 	for i in {1..2}; do \
# 		echo "Running iteration $$i"; \
# 		python3 partA.py train --model "model_ensembles$$i" --num_ensembles 10 --epochs 1000 --seed $${seeds[$$i-1]}; \
# 	done

# plot_10_models_with_ensembles:
# 	#make list of seeds	
# 	seeds=(123 234 345 456 567 678 789 890 901 12); \
# 	for i in {1..1}; do \
# 		echo "Running iteration $$i"; \
# 		python3 partA.py plot --model "model_ensembles$$i" --num_ensembles 10 --epochs 1000 --seed $${seeds[$$i-1]}; \
# 	done

# plot_10_models_without_ensembles:
# 	#make list of seeds	
# 	seeds=(123 234 345 456 567 678 789 890 901 12); \
# 	for i in {1..1}; do \
# 		echo "Running iteration $$i"; \
# 		python3 partA.py plot --model "model$$i" --num_ensembles 1 --epochs 100 --seed $${seeds[$$i-1]}; \
# 	done