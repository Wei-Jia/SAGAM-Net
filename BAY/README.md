## Data Preparation

The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), and should be
put into the `data/` folder.
The `*.h5` files store the data in `panads.DataFrame` using the `HDF5` file format. Here is an example:

|                     | sensor_0 | sensor_1 | sensor_2 | sensor_n |
| :-----------------: | :------: | :------: | :------: | :------: |
| 2018/01/01 00:00:00 |   60.0   |   65.0   |   70.0   |   ...    |
| 2018/01/01 00:05:00 |   61.0   |   64.0   |   65.0   |   ...    |
| 2018/01/01 00:10:00 |   63.0   |   65.0   |   60.0   |   ...    |
|         ...         |   ...    |   ...    |   ...    |   ...    |

Here is an article about [Using HDF5 with Python](https://medium.com/@jerilkuriakose/using-hdf5-with-python-6c5242d08773).

Run the following commands to generate train/test/val dataset at `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.

