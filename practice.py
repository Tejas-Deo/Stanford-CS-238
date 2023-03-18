import time


start_time = time.time()
print("Start time: ", start_time)

while True:
    current_time = time.time()

    if current_time - start_time > 5:
        print("Value of subtraction: ", current_time - start_time)
        print("Exiting loop: ")
        break


print("after while loop")