from setuptools import setup
import os
from glob import glob

package_name = 'graph_datasets'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_data={
        package_name + '.config': ['*.json'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),glob(os.path.join('config', '*.json')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='josmilrom@gmail.com',
    license='TODO: License declaration',
    tests_require=['pytest']
)