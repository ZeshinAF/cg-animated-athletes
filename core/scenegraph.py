"""Hierarchical (scene-graph) model of an articulated body.

Each Node stores a fixed offset from its parent joint and a list of joint
rotations whose angles are read from the current *pose* (a dict
joint-name -> degrees).  The world matrix of a node is

    W_node = W_parent * T(offset) * R_1(pose[j1]) * R_2(pose[j2]) ...

The same matrices are used for rendering (glMultMatrixf) and for forward
kinematics, e.g. to find where the hand is so the ball can follow it.
"""
from OpenGL.GL import glMultMatrixf, glPopMatrix, glPushMatrix

from .mathutil import identity, rotate, to_gl, translate


class Node:
    def __init__(self, name, offset=(0.0, 0.0, 0.0), joints=(), draw=(), children=()):
        self.name = name
        self.offset = offset          # tuple or callable(pose) -> (x, y, z)
        self.joints = list(joints)    # [(axis, pose_key, sign), ...]
        self.draw = list(draw)        # callables drawing geometry in local space
        self.children = list(children)
        self.local = identity()
        self.world = identity()

    def add(self, *children):
        self.children.extend(children)
        return self

    def local_matrix(self, pose):
        off = self.offset(pose) if callable(self.offset) else self.offset
        m = translate(*off)
        for axis, key, sign in self.joints:
            angle = pose.get(key, 0.0)
            if angle:
                m = m @ rotate(axis, sign * angle)
        return m

    def update(self, parent_world, pose):
        self.local = self.local_matrix(pose)
        self.world = parent_world @ self.local
        for c in self.children:
            c.update(self.world, pose)

    def render(self):
        glPushMatrix()
        glMultMatrixf(to_gl(self.local))
        for d in self.draw:
            d()
        for c in self.children:
            c.render()
        glPopMatrix()

    def find(self, name):
        if self.name == name:
            return self
        for c in self.children:
            hit = c.find(name)
            if hit is not None:
                return hit
        return None

    def world_pos(self, local_point=(0.0, 0.0, 0.0)):
        x, y, z = local_point
        return (self.world @ [x, y, z, 1.0])[:3]
