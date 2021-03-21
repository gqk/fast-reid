.EXPORT_ALL_VARIABLES:

.PHONEY: market dukemtmc msmt

market: CUDA_VISIBLE_DEVICES=0
market:
	@echo "Target: market"
	@echo "Envs:   CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"
	@nohup python tools/train_net.py \
		--config-file configs/Market1501/bagtricks_R50.yml \
		SOLVER.FP16_ENABLED False \
		> market.out 2>&1 &

define market_inc
	$(eval stage := $1)
	$(eval stage_pre := $2)

	@echo "Target: market_inc_$(stage)"
	@echo "Envs:   CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"

	@nohup python tools/train_net.py \
		--config-file configs/Market1501/bagtricks_R50.yml \
		DATASETS.SPLITNO $(stage) \
		OUTPUT_DIR logs/market1501/bagtricks_R50/stage_$(stage) \
		MODEL.WEIGHTS logs/market1501/bagtricks_R50/stage_$(stage_pre)/model_best.pth \
		SOLVER.FP16_ENABLED False \
		SOLVER.BASE_LR 0.000035 \
		SOLVER.WARMUP_ITERS 0 \
		SOLVER.STEPS [] \
		> market_inc_$(stage).out 2>&1 &
endef

market_inc_0: CUDA_VISIBLE_DEVICES=0
market_inc_0:
	@echo "Target: market_inc_0"
	@echo "Envs:   CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"
	@nohup python tools/train_net.py \
		--config-file configs/Market1501/bagtricks_R50.yml \
		DATASETS.SPLITNO 0 \
		OUTPUT_DIR logs/market1501/bagtricks_R50/stage_0 \
		SOLVER.FP16_ENABLED False \
		> market_inc_0.out 2>&1 &

market_inc_1: CUDA_VISIBLE_DEVICES=0
market_inc_1:
	$(call market_inc,1,0)

market_inc_2: CUDA_VISIBLE_DEVICES=0
market_inc_2:
	$(call market_inc,2,1)

market_inc_3: CUDA_VISIBLE_DEVICES=0
market_inc_3:
	$(call market_inc,3,2)

market_inc_4: CUDA_VISIBLE_DEVICES=0
market_inc_4:
	$(call market_inc,4,3)

market_inc_5: CUDA_VISIBLE_DEVICES=0
market_inc_5:
	$(call market_inc,5,4)

################################################################################

dukemtmc: CUDA_VISIBLE_DEVICES=0
dukemtmc:
	@echo "Target: dukemtmc"
	@echo "Envs:   CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"
	@nohup python tools/train_net.py \
		--config-file configs/DukeMTMC/bagtricks_R50.yml \
		SOLVER.FP16_ENABLED False \
		> dukemtmc.out 2>&1 &

define dukemtmc_inc
	$(eval stage := $1)
	$(eval stage_pre := $2)

	@echo "Target: dukemtmc_inc_$(stage)"
	@echo "Envs:   CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"

	@nohup python tools/train_net.py \
		--config-file configs/DukeMTMC/bagtricks_R50.yml \
		DATASETS.SPLITNO $(stage) \
		OUTPUT_DIR logs/dukemtmc/bagtricks_R50/stage_$(stage) \
		MODEL.WEIGHTS logs/dukemtmc/bagtricks_R50/stage_$(stage_pre)/model_best.pth \
		SOLVER.FP16_ENABLED False \
		SOLVER.BASE_LR 0.000035 \
		SOLVER.WARMUP_ITERS 0 \
		SOLVER.STEPS [] \
		> dukemtmc_inc_$(stage).out 2>&1 &
endef

dukemtmc_inc_0: CUDA_VISIBLE_DEVICES=0
dukemtmc_inc_0:
	@echo "Target: dukemtmc_inc_0"
	@echo "Envs:   CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"
	@nohup python tools/train_net.py \
		--config-file configs/DukeMTMC/bagtricks_R50.yml \
		DATASETS.SPLITNO 0 \
		OUTPUT_DIR logs/dukemtmc/bagtricks_R50/stage_0 \
		SOLVER.FP16_ENABLED False \
		> dukemtmc_inc_0.out 2>&1 &

dukemtmc_inc_1: CUDA_VISIBLE_DEVICES=0
dukemtmc_inc_1:
	$(call dukemtmc_inc,1,0)

dukemtmc_inc_2: CUDA_VISIBLE_DEVICES=0
dukemtmc_inc_2:
	$(call dukemtmc_inc,2,1)

dukemtmc_inc_3: CUDA_VISIBLE_DEVICES=0
dukemtmc_inc_3:
	$(call dukemtmc_inc,3,2)

dukemtmc_inc_4: CUDA_VISIBLE_DEVICES=0
dukemtmc_inc_4:
	$(call dukemtmc_inc,4,3)

dukemtmc_inc_5: CUDA_VISIBLE_DEVICES=0
dukemtmc_inc_5:
	$(call dukemtmc_inc,5,4)

################################################################################

msmt17: CUDA_VISIBLE_DEVICES=0
msmt17:
	@echo "Target: msmt17"
	@echo "Envs:   CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"
	@nohup python tools/train_net.py \
		--config-file configs/MSMT17/bagtricks_R50.yml \
		SOLVER.FP16_ENABLED False \
		> dukemtmc.out 2>&1 &

define msmt17_inc
	$(eval stage := $1)
	$(eval stage_pre := $2)

	@echo "Target: msmt17_inc_$(stage)"
	@echo "Envs:   CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"

	@nohup python tools/train_net.py \
		--config-file configs/MSMT17/bagtricks_R50.yml \
		DATASETS.SPLITNO $(stage) \
		OUTPUT_DIR logs/msmt17/bagtricks_R50/stage_$(stage) \
		MODEL.WEIGHTS logs/msmt17/bagtricks_R50/stage_$(stage_pre)/model_best.pth \
		SOLVER.FP16_ENABLED False \
		SOLVER.BASE_LR 0.000035 \
		SOLVER.WARMUP_ITERS 0 \
		SOLVER.STEPS [] \
		> msmt17_inc_$(stage).out 2>&1 &
endef


msmt17_inc_0: CUDA_VISIBLE_DEVICES=0
msmt17_inc_0:
	@echo "Target: msmt17_inc_0"
	@echo "Envs:   CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"
	@nohup python tools/train_net.py \
		--config-file configs/MSMT17/bagtricks_R50.yml \
		DATASETS.SPLITNO 0 \
		OUTPUT_DIR logs/msmt17/bagtricks_R50/stage_0 \
		SOLVER.FP16_ENABLED False \
		> msmt17_inc_0.out 2>&1 &

msmt17_inc_1: CUDA_VISIBLE_DEVICES=0
msmt17_inc_1:
	$(call msmt17_inc,1,0)

msmt17_inc_2: CUDA_VISIBLE_DEVICES=0
msmt17_inc_2:
	$(call msmt17_inc,2,1)

msmt17_inc_3: CUDA_VISIBLE_DEVICES=0
msmt17_inc_3:
	$(call msmt17_inc,3,2)

msmt17_inc_4: CUDA_VISIBLE_DEVICES=0
msmt17_inc_4:
	$(call msmt17_inc,4,3)

msmt17_inc_5: CUDA_VISIBLE_DEVICES=0
msmt17_inc_5:
	$(call msmt17_inc,5,4)
