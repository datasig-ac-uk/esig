import sys
import numpy as np
from numpy.linalg import norm
import time
import esig.tosig as ts
import esig


def test ():
   failures = 0

   dimension = 60 ## big test set this to 2400 small test 240
   no_points = 100000

   print("preparing data ({},{})".format(no_points,dimension))
   tic = time.perf_counter()
   np.random.seed(0)
   data = np.random.rand(no_points,dimension)
   toc = time.perf_counter()
   print("prepared data in {n:0.4f} seconds\n".format(n=toc - tic))

   ## test 1
   tic = time.perf_counter()
   selected_points, new_weights = ts.recombine(data) ## degree = 1
   toc = time.perf_counter()
   print("Recombined {no_points} points in dimension {dimension} to {dimension2} points in {n:0.4f} seconds".format(no_points=no_points,dimension=dimension,dimension2=dimension+1,n=toc - tic))

   ## check mean preserved
   old_average=np.sum(data, 0)
   new_average=new_weights.dot(np.take(data, selected_points, 0))
   normalised_error = norm(old_average - new_average)/(norm(old_average) + norm(new_average))
   print("normalised difference in the integrals = ",normalised_error)

   ## report
   if ((len(selected_points) > dimension + 1) or (normalised_error > 1e-13)):
      failures = failures + 1
      print("test 1 failed")
   else:
      print("test 1 passed")

   ## test2
   ### the points are not spanning the full space and so the minimal set should have cardinality less than or equal rank + 1
   matrix = np.random.rand(dimension,dimension + 20)
   new_data = data.dot(matrix)
   tic = time.perf_counter()
   selected_points, new_weights = ts.recombine(new_data) ## degree = 1
   toc = time.perf_counter()
   print("\nRecombined {no_points} points in dimension {dimension} to {dimension2} points in {n:0.4f} seconds".format(no_points=no_points,dimension=dimension + 20,dimension2=dimension+1,n=toc - tic))

   ## check mean preserved
   old_average=np.sum(data, 0)
   new_average=new_weights.dot(np.take(data, selected_points, 0))
   normalised_error = norm(old_average - new_average)/(norm(old_average) + norm(new_average))
   print("normalised difference in the integrals = ",normalised_error)
   print("no of points left = ", len(selected_points))

   ## report
   if ((len(selected_points) > dimension + 1) or (normalised_error > 1e-12)):
      failures = failures + 1
      print("test 2 failed")
   else:
      print("test 2 passed")

   #test3
   ## test the degree > 1 case - match second moments
   dimension = 10
   no_points = 1000
   data = np.random.rand(no_points,dimension)

   tic = time.perf_counter()
   selected_points, new_weights = ts.recombine(data, degree = 2)
   toc = time.perf_counter()
   print("\nRecombined {no_points} points in dimension {dimension} to {m} points in {n:0.4f} seconds while preserving first and second moments".format(no_points=no_points,dimension=dimension,m=len(selected_points),n=toc - tic))

   old_average=np.sum(data, 0)
   new_average=new_weights.dot(np.take(data, selected_points, 0))
   normalised_error_in_mean = norm(old_average - new_average)/(norm(old_average) + norm(new_average))

   new_cov = np.cov(np.take(data, selected_points, 0),rowvar=False, bias=True, aweights=new_weights)
   old_cov = np.cov(data,rowvar=False, bias=True,aweights=np.full(1000,1.))
   normalised_error_in_cov = norm(old_cov - new_cov)/(norm(old_cov) + norm(new_cov))

   print ("normalised_error_in_mean {}\nnormalised_error_in_cov  {}".format(normalised_error_in_mean,normalised_error_in_cov))

   if ((normalised_error_in_mean > 1e-13) or (normalised_error_in_cov > 1e-13)):
      failures = failures + 1
      print("test 3 failed")
   else:
      print("test 3 passed")

   print ("\n")

   import platform, subprocess

   print("="*40, "System Information", "="*40)
   uname = platform.uname()
   print("System: {}".format(uname.system))
   print("Node Name: {}".format(uname.node))
   print("Release: {}".format(uname.release))
   print("Version: {}".format(uname.version))
   print("Machine: {}".format(uname.machine))
   print("Processor: {}".format(uname.processor))

   return failures
