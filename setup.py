import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='pose_estimation_mp',  
     version='0.1',
     scripts=['pose_estimation_mp'] ,
     author="Samar Kale",
     author_email="rz.samar.kale@gmail.com",
     description="A Body, Hand, Face tracking utility",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url=" https://github.com/REZ3LIET/Pose_Estimations",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )