''' this script is taken from https://www.youtube.com/watch?v=oXWVuK_NjbY '''

import os
import tensorflow as tf

# get the two numbers to add from the command prompt
intNum1 = int(raw_input('enter num 1: '))
intNum2 = int(raw_input('enter num 2: '))

# establish two tensors, one for each input number
num1 = tf.Variable(intNum1, name='num1')
num2 = tf.Variable(intNum2, name='num2')

#establish garph
sum = tf.add(num1,num2, name='sum')

# notes that this shows informationa bout sum, but does NOT evaluate anything yet
print ('sum - ' + str(sum))

# instantiate a global variables initializer
globalVarsInitializer = tf.global_variables_initializer()

# finally we can run the graph (in a session)
with tf.Session() as sess:
    globalVarsInitializer.run()
    result = sum.eval()
# end with

# show the result
print('result - ' + str(result))

# write the graph to file so we can view with TensorBoard
fileWriter = tf.summary.FileWriter(os.getcwd())
fileWriter.add_graph(sess.graph)
fileWriter.close()


