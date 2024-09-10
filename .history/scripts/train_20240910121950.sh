# example to run:
#       bash train.sh PERACT_BC 0,1 12345 ${exp_name}

# set the method name
method=${1} # PERACT_BC or BIMANUAL_PERACT

# set the seed number
seed="0"

train_gpu=${2:-"0,1"}
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
port=${3:-"12345"}
# you could enable/disable wandb by this.
use_wandb=True

# cur_dir=$(pwd)
train_demo_path="/mnt/disk_1/tengbo/bimanual_data/train"

# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
exp_name=${4:-"${method}_${addition_info}"}
logdir="/mnt/disk_1/tengbo/peract_bimanual/log"
instructions="/mnt/disk_1/tengbo/peract_bimanual/18_lang_template.pkl"
# replay_path="/mnt/disk_1/tengbo/replay/"

# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (5s)"
sleep 1s
tmux kill-session -t ${exp_name}
sleep 3s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}

#######
# override hyper-params in config.yaml
#######
# batch_size=2
# skill_predictor=False

batch_size=2
skill_predictor=True

# task_name=${"multi_${addition_info}"}


######## Revise frequently
load_existing_weights=False
use_pre=True
timesteps=1
# 13 tasks in total, without (e)put_item_in_drawer now
tasks=[bimanual_pick_laptop,bimanual_pick_plate,bimanual_straighten_rope,coordinated_lift_ball,coordinated_lift_tray,coordinated_push_box,coordinated_put_bottle_in_fridge,dual_push_buttons,handover_item,bimanual_sweep_to_dustpan,coordinated_take_tray_out_of_oven,handover_item_easy]
# # bimanual_sweep_to_dustpan,coordinated_take_tray_out_of_oven,handover_item_easy
# tasks=[coordinated_push_box]
demo=100
episode_length=25
save_freq=10000
log_freq=100
task_folder="multi"
replay_path="/mnt/disk_2/tengbo/replay_/"



# for debug
# tasks=[bimanual_pick_laptop,bimanual_pick_plate,bimanual_straighten_rope,coordinated_lift_ball,coordinated_lift_tray,coordinated_push_box,coordinated_put_bottle_in_fridge,dual_push_buttons,handover_item,bimanual_sweep_to_dustpan,coordinated_take_tray_out_of_oven,handover_item_easy]
# tasks=[dual_push_buttons]
# demo=1
# episode_length=4
# save_freq=100
# log_freq=1
# wandb_project="debug"
# task_folder="debug"
# replay_path="/mnt/disk_2/tengbo/replay_/debug/"
#########

tmux select-pane -t 0 
tmux send-keys "conda activate per2; 
CUDA_VISIBLE_DEVICES=${train_gpu} python train.py method=$method \
        rlbench.task_name=${exp_name} \
        framework.logdir=${logdir} \
        rlbench.demo_path=${train_demo_path} \
        framework.start_seed=${seed} \
        framework.use_wandb=${use_wandb} \
        framework.wandb_group=${exp_name} \
        framework.wandb_name=${exp_name} \
        ddp.num_devices=${#train_gpu_list[@]} \
        replay.batch_size=${batch_size} \
        ddp.master_port=${port} \
        rlbench.tasks=${tasks} \
        rlbench.demos=${demo} \
        rlbench.episode_length=${episode_length} \
        framework.save_freq=${save_freq} \
        framework.load_existing_weights=${load_existing_weights} \
        framework.wandb_project=${wandb_project} \
        framework.use_pretrained=${use_pre} \
        framework.use_predictor=${skill_predictor} \
        replay.path=${replay_path} \
        replay.task_folder=${task_folder} \
        rlbench.instructions=${instructions} \
        framework.log_freq=${log_freq} \
        replay.timesteps=${timesteps}
"
# remove 0.ckpt
# rm -rf logs/${exp_name}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}