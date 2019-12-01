import sys
import os
import os.path


def sort_list(j):
    if(j>=0 and j<=9):
        n=2
    elif (j>=10 and j<=99):
        n=1
    elif (j>=100 and j<=999):
        n=0
    else:
        print('too many image data, more than 1000.')
    return n

if __name__ == '__main__':
    obj_class = sys.argv[1]
    file_type = '.jpg'
    parent_file_path = './orig_objs/'
    child_file_path = [obj_class+'/']
    new_name_list = [obj_class]
    # new_name_list = ['aaa']
    old_data_num = 0
    
    try:
        for i,item in enumerate(child_file_path):
            old_data_num = 0
            new_data_num = 0
            file_path = parent_file_path + item
            file_name_list = os.listdir(file_path)
            file_name_list.sort()
            print('file_name_list len', len(file_name_list))
            
            std_file_name_len = len(obj_class + '000.jpg')
            for (j, file_name) in enumerate(file_name_list):
                if(new_name_list[i] in file_name)and(len(file_name)<=std_file_name_len):
                    old_data_num+=1
                else:
                    new_data_num+=1
                    rand_name = 'zzz_'+str(new_data_num)    # Do not change 'zzz' this is set for sort (z is the final ranking. sort rank: 0 -> A-Z -> a-z)
                    os.rename(file_path+file_name, file_path+rand_name)

            print('new_data_num = ', new_data_num)
            file_name_list = os.listdir(file_path)
            file_name_list.sort()
            # 
            
            for (j, file_name) in enumerate(file_name_list):
                if(j < old_data_num):
                    continue
                # print([j, old_data_num])
                redun_0 = '0' * sort_list(j)
                # new_name = (new_name_list[i] + redun_0 + str(j+1) + file_type)
                new_name = (new_name_list[i] + str(j+1) + file_type)

                old_name = file_path + file_name
                new_name = file_path + new_name
                os.rename(old_name, new_name)
    except:
        print('no such folder:', parent_file_path+item)
