from setuptools import setup
# from setuptools import find_packages
import os
from glob import glob

package_name = 'gyrobro_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # (os.path.join('share', package_name), glob('urdf/*')),
        ('share/' + package_name+'/urdf/', glob('urdf/*')),
        ('share/' + package_name+'/rviz/', glob('rviz/*')),
        ('share/' + package_name+'/meshes/', glob('meshes/*')),
        # (os.path.join('share', package_name), glob('meshes/*')),
        # (os.path.join('share', package_name), glob('rviz/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoggi',
    maintainer_email='vldanilov90@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    
    entry_points={
        'console_scripts': [
            'pybullet = gyrobro_sim.gyrobro_pybullet:main',
        ],
    },
)
