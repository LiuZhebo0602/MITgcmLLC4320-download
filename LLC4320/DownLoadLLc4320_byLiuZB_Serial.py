# from aiohttp import ClientResponseError
# from concurrent.futures import ThreadPoolExecutor
from xmitgcm import llcreader
import os
import dask
import xarray as xr
import time
# 配置dask，设置切分大块数组的策略为不切分
dask.config.set(**{'array.slicing.split_large_chunks': False})
import numpy as np
# import concurrent.futures
from datetime import datetime
import pandas as pd

'''
created by LiuZhenbo in 2023.11.30
这个版本的代码是串行下载LLC4320模型数据的代码，可以根据需要修改参数和下载时间范围
具体数据说明按LLC4320文件说明.ipynb文件中的说明
同时在该文件中说明了该数据下载code的下一步修改方向
1: Face 参数：face 值为 3，是手动设置的。可以修改为动态输入，或根据需要在代码顶部或配置文件中定义。
2: 参数配置：每个 face 的 i_start、i_end、j_start、j_end 区域参数是需要给定的。如果面数和范围经常改变，可以考虑改用配置文件或传入参数列表，使代码更灵活。
这里因为是为了下载固定区域的数据，所以给定了不同的face的i_start、i_end、j_start、j_end参数。实际使用中可以自定义。
3: 并行下载：当前代码只使用串行下载，可以考虑恢复被注释掉的并行下载部分，例如 ThreadPoolExecutor，以提升下载效率。
4: 日志记录：下载进度记录在 Face{face}_download_progress.txt 中。如果文件路径需要变化，可以将其设为可配置的变量。
5: 下载变量：当前下载了特定的变量（U, V, Salt, Theta, Eta）。如果想要下载不同的变量，需在代码中更新 varnames 参数，或将变量名列表改为参数输入。 ​​
'''

def download_and_monitor(ds, hour, params, max_retries=10, timeout_minutes=10):
    # 提取参数中的'face'值
    face = params['face']
    # 记录开始时间
    start_time = datetime.now()
    try:
        # 尝试多次下载，最大重试次数为max_retries
        for attempt in range(max_retries):
            try:
                # 检查并创建文件夹，以便存放下载结果
                folder_name = f'MITGCM_llc4320_Arabian_Sea_Face_{face}'
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # 定义要提取数据的区域，使用切片
                # region_slice = {'i':slice(0,5800-4320),'j':slice(3300,4320),'i_g':slice(0,5800-4320),'j_g':slice(3300,4320),'face':face-1,'time':slice(hour,hour+1)}
                region_slice = {'i':slice(params['i_start'], params['i_end']),'j':slice(params['j_start'], params['j_end']),
                                'i_g':slice(params['i_start'], params['i_end']),'j_g':slice(params['j_start'], params['j_end']),
                                'face':params['face']-1,'time':slice(hour,hour+1)}

                # 根据定义的切片从数据集中选择数据
                region = ds.isel(**region_slice)

                # 创建新的xarray数据集来存储所需变量
                ds_new = xr.Dataset()
                ds_new['U'] = xr.DataArray(region.U.isel(time=0).data, dims=('k', 'i', 'j'))  # 按实际情况设置维度名
                ds_new['V'] = xr.DataArray(region.V.isel(time=0).data, dims=('k','i', 'j'))
                ds_new['Salt'] = xr.DataArray(region.Salt.isel(time=0).data, dims=('k','i', 'j'))
                # ds_new['W'] = xr.DataArray(region.W.isel(time=0).data, dims=('k','i', 'j')) # 我没有下载w，所以注释掉
                ds_new['Theta'] = xr.DataArray(region.Theta.isel(time=0).data, dims=('k','i', 'j'))
                ds_new['Eta'] = xr.DataArray(region.Eta.isel(time=0).data, dims=('i', 'j'))
                # 增加时间信息到数据集
                ds_new['time'] = str(region.time.data[0])[:13]
                # break  # 如果成功执行，跳出循环
                filename = f'{folder_name}/{hour}_Face{face}_' + str(region.time.data[0])[:13] + '.nc'
                # 如果文件存在，删除旧文件
                if os.path.exists(filename):
                    os.remove(filename)
                # 将新数据集保存为netCDF文件
                ds_new.to_netcdf(filename)

                # 记录结束时间，计算持续时间
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                # 返回下载成功信息
                return hour, True, duration, None

            except Exception as e:
                # 如果出错，显示错误信息并重试
                print(f'Error at hour {hour}, attempt {attempt + 1}: {e}')
                filename = f'{folder_name}/{hour}_Face{face}_' + str(region.time.data[0])[:13] + '.nc'
                if os.path.exists(filename):
                    os.remove(filename)
                # 等待60秒后再次尝试
                time.sleep(60)  # 等待60秒后重试

        else:
            # 重试多次失败后，返回失败信息
            # with open(f'Face{face}_download_progress.txt', 'a') as progress_file:
            #     progress_file.write(f'{hour} Failed to download for face {face}\n')
            filename = f'{folder_name}/{hour}_Face{face}_' + str(region.time.data[0])[:13] + '.nc'
            if os.path.exists(filename):
                os.remove(filename)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return hour, False, duration, str(e)
    except Exception as e:
        # 捕获意外异常并记录时间信息
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        return hour, True, duration, None


# def update_excel(record, excel_path):
#     df = pd.DataFrame(columns=["Hour", "Download Completed", "Duration", "Error"])
#     if os.path.exists(excel_path):
#         df = pd.read_excel(excel_path, index_col=0)

#     # 更新或添加新记录
#     df.loc[record[0]] = record[1:]

#     # 保存到Excel文件
#     df.to_excel(excel_path)

def download_data(ds, start_hour, end_hour, params):
    # 从参数中提取'face'值
    face = params['face']

    # 定义要下载的小时范围
    # hours_to_download = np.arange(start_hour, end_hour)
    # excel_path = f"Face{face}_download_progress.xlsx"

    # 使用多线程实现并行下载
    # # 使用 ThreadPoolExecutor 实现并行下载
    # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #     future_to_hour = {executor.submit(download_and_monitor,ds, hour, params): hour for hour in hours_to_download}
    #     for future in concurrent.futures.as_completed(future_to_hour):
    #         hour, success = future.result()
    #         if success:
    #             print(f"Successfully downloaded hour: {hour}")
    #             with open(f'Face{face}_download_progress.txt', 'a') as progress_file:
    #                 progress_file.write(f'{hour} Downloaded for face {face}\n')
    #         else:
    #             print(f"Failed to download hour: {hour}")
    #             with open(f'Face{face}_download_progress.txt', 'a') as progress_file:
    #                 progress_file.write(f'{hour} Failed to download for face {face}\n')
    
    # 串行处理多个下载任务
    for hour in range(start_hour, end_hour):
        No_hour, status, duration, reason = download_and_monitor(ds=ds, hour=hour, params=params)
        # 将下载结果写入进度文件
        with open(f'Face{face}_download_progress.txt', 'a') as progress_file:
                progress_file.write(f'{No_hour}, {str(status)}, {duration}s, {reason}\n')
        # update_excel(record, excel_path)

if __name__ == '__main__':
    # 使用 llcreader 从 ECCO 门户获取 LLC4320 模型数据
    model = llcreader.ECCOPortalLLC4320Model()
    # ！！！！定义要读取的层数，这里指的是下载0-29层（共30层）！！！！
    k_levels = list(range(0, 30))
    # ！！！！获取数据集，包括特定变量和深度层！！！！
    ds = model.get_dataset(varnames=['U', 'V', 'Salt', 'Theta', 'Eta'], k_levels=k_levels)
    # 这里需要下载的变量，请务必和53-58行的变量列表保持一致，如果添加或删除变量，请同步修改。比如这里添加了W变量
    # varnames=['U', 'V', 'Salt', 'Theta', 'Eta', 'W']，请取消注释56行的W变量下载代码

    face = 3  # 你的 face 值，可以根据需要设置，这里设置为3，请注意！！！为了和LLC4320文件说明.ipynb开头给出的示意图一致
    # 这里的face=3，就是图上的face3。虽然在实际下载过程中，下载图中的face3需要给的是face=2
    # 46行我手动把face-1


    # 从什么开始时间，什么结束时间下载数据
    start_time = 0
    end_time = 3000

    # 初始化参数字典
    params = {
        'face': None,
        'i_start': None,
        'i_end': None,
        'j_start': None,
        'j_end': None
    }

    # 根据 face 值设置参数
    # 为每个 face 值设置不同的参数
    # ！！！！如果需要下载的区域不是这个区域，可以根据需要修改这里的参数！！！！
    # 但是实际上调用的face是155行的face，记得同步修改
    if face == 3:
        params['face'] = 3
        params['i_start'] = 0
        params['i_end'] = 1480
        params['j_start'] = 1000
        params['j_end'] = 4320
    elif face == 8:
        params['face'] = 8
        params['i_start'] = 0
        params['i_end'] = 222
        params['j_start'] = 0
        params['j_end'] = 333
    else:
        # 如果 face 不匹配任何条件，抛出异常
        raise ValueError('Invalid face value')

    # 开始下载数据，使用定义好的参数和时间列表
    download_data(ds=ds, start_hour=start_time, end_hour=end_time, params=params)