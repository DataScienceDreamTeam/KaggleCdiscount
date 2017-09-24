import multiprocessing as mp
import time
import itertools

def Func(N):
    print("in :" + str(N))
    time.sleep(2)

if __name__ == '__main__':

    myRange = range(100)

    mySlice1 =  itertools.islice(myRange, 0, 20)
    mySlice2 =  itertools.islice(myRange,20, 40)
    mySlice3 =  itertools.islice(myRange,40, 60)
    mySlice4 =  itertools.islice(myRange,60, 80)
    mySlice5 =  itertools.islice(myRange,80, 110)

    for i,j in enumerate(mySlice1):
        print(str(i) + " " + str(j))
    print("ha")
    for i,j in enumerate(mySlice5):
        print(str(i) + " " + str(j))

    # print("start")
    # pool=mp.Pool(2)
    # pool.map_async(Func,range(100))

    # print("texte 1")
    # print("texte 2")
    # print("texte 3")
    # print("end")

    # pool.close()
    # pool.join()