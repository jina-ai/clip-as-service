#nohup python toy2.py > logs/clip_console.log 2>&1 &
#nohup python toy1.py > logs/ranker_console.log 2>&1 &

Nproc=50

for((i=1;i<=$Nproc;i++));do
    nohup python toy3.py proc-$i > logs/stress_console.log.$i 2>&1 &
done