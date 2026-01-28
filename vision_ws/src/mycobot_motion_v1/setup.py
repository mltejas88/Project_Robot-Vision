from setuptools import find_packages, setup

package_name = 'mycobot_motion_v1'

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
    maintainer='tejas',
    maintainer_email='your.email@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motion_node = mycobot_motion_v1.motion_node:main',
            'motion_node_v1 = mycobot_motion_v1.motion_node_v1:main',
            'motion_node_v2 = mycobot_motion_v1.motion_node_v2:main',
            'motion_node_v4 = mycobot_motion_v1.motion_node_v4:main',
            'motion_node_v3 = mycobot_motion_v1.motion_node_v3:main',
        ],
    },
)
