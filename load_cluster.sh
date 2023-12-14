
#! /bin/bash

scp -P 2222 -i ~/cluster_key ~/PycharmProjects/GeometricRL/hyperMaze/* areichlin@moria.csc.kth.se:~/GeometricRL/hyperMaze/
scp -P 2222 -i ~/cluster_key ~/PycharmProjects/GeometricRL/mazes/* areichlin@moria.csc.kth.se:~/GeometricRL/mazes/
scp -P 2222 -i ~/cluster_key ~/PycharmProjects/GeometricRL/fetch/* areichlin@moria.csc.kth.se:~/GeometricRL/fetch/
scp -P 2222 -i ~/cluster_key ~/PycharmProjects/GeometricRL/models/* areichlin@moria.csc.kth.se:~/GeometricRL/models/
scp -P 2222 -i ~/cluster_key ~/PycharmProjects/GeometricRL/my_utils/* areichlin@moria.csc.kth.se:~/GeometricRL/my_utils/