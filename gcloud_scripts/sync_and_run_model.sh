#!/bin/bash

COMMAND="python3.6 $1 -s $2"

NB_INSTANCES=${3:-1}
echo $NB_INSTANCES
echo "START SCRIPT"

echo "Dataset: $1 and setting file: $2"

make clean
echo "LOCAL: folders cleaned"

iter=$NB_INSTANCES
for v in $(seq 1 $NB_INSTANCES)
do
    INSTANCE_NAME=$(grep GCLOUD_INSTANCE_NAME_$v .env | cut -d '=' -f2)
    ZONE_NAME=$(grep GCLOUD_INSTANCE_ZONE_$v .env | cut -d '=' -f2)
    echo $INSTANCE_NAME
    echo $ZONE_NAME

    gcloud compute instances start $INSTANCE_NAME --zone=$ZONE_NAME
    returned=$(echo $?)
    if [[ $returned == 0 ]]
    then
        echo "VM: started"
        notify-send "DeepSwarm: Nohup model launched"
        sleep 20 #otherwise, ssh connection refused
        gcloud compute scp --recurse deepswarm/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp --recurse examples/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp --recurse settings/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp --recurse tests/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp requirements-dev-gpu.txt $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp Makefile $INSTANCE_NAME:~ --zone=$ZONE_NAME
        echo "VM: data transfered from local to instance"

        # gcloud compute ssh --zone=$ZONE_NAME $INSTANCE_NAME --command 'python3.6 -m pip install -r requirements-dev.txt'
        gcloud compute ssh --zone=$ZONE_NAME $INSTANCE_NAME --command "$COMMAND" 
        echo "VM: commands executed"
        
        gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE_NAME

        notify-send "DeepSwarm: Nohup model launched"
        echo "END SCRIPT"
        exit
    fi
done 