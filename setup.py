from setuptools import setup, find_packages

package_name = 'person_tracking_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='landsys',
    maintainer_email='landsys@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_tracker = person_tracking_ros.person_tracker:main',
            'deep_sort = person_tracking_ros.deep_sort_node:main'
        ],
    },
)
