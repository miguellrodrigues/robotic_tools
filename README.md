
# Robotic Tools

Robotic tools is a library made to make some calculations easier, like robots forward
kinematic's and dynamics. There is also an numerical implementation of inverse velocity kinematic's.

You can use this lib for any robot, since you have the Denavit Hartenberg parameters.

## Forward Kinematics

in order to use the forward kinematics, you gonna need the robot DH parameters. Then
u can create a 'Link' object representation for each link, using the parameters.

```python
L0 = Link([θ1, d1, a1, α1])
L1 = Link([θ2, d2, a2, α2])
L2 = Link([θ3, d3, a3, α3])
```

Finally create an instance of the ForwardKinematic class, and pass a list with
all links in the constructor. You can also pass an offset with the angles of home position.

```python
fk = ForwardKinematic([L0, L1, L2], offset=np.array([.0, .0, .0]))
```
