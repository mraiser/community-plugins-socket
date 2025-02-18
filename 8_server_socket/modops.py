#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import io
import base64
import imutils as imutils

from .abstractop import AbstractOp
from core import G
import mh
import json


class SocketModifierOps(AbstractOp):

    def __init__(self, sockettaskview):
        super().__init__(sockettaskview)
        self.functions["applyModifier"] = self.applyModifier
        self.functions["getAppliedTargets"] = self.getAppliedTargets
        self.functions["getAvailableModifierNames"] = self.getAvailableModifierNames
        self.functions["getModifierValue"] = self.getModifierValue
        self.functions["setTarget"] = self.setTarget
        self.functions["snapshot"] = self.snapshot
        self.functions["resetCamera"] = self.resetCamera
        self.functions["maximizeWindow"] = self.maximizeWindow
        self.functions["landmarks"] = self.landmarks
        self.functions["approachTarget"] = self.approachTarget
        self.functions["optimizeTargetRotation"] = self.optimizeTargetRotation
        self.target = None
        self.target_image = None
        self.target_rotation = 0.0

    def optimizeTargetRotation(self, conn, jsonCall):
        keys = self.target.keys()
        data = self.target
        nudata = self.analyze()
        delta = self.calcdelta(data, nudata, keys)
        x = data['X_4']
        y = data['Y_4']
        bestrot = self.target_rotation
        img = self.target_image
        newimg = self.rotateImg(img, self.target_rotation, x, y)
        for i in range(-100, 100, 1):
            tmpimg = self.rotateImg(img, i / 10, x, y)
            tmp = self.getLandmarks(tmpimg)
            if len(tmp.keys()) > 0:
                nudelta = self.calcdelta(tmp, nudata, keys)
                if (nudelta < delta):
                    data = tmp
                    delta = nudelta
                    bestrot = i / 10
                    newimg = tmpimg

        print('BEST ROTATION IS: ' + str(bestrot))
        self.target = data
        self.target_rotation = bestrot

        img = newimg
        shape = img.shape
        img = img.reshape(-1)
        img = base64.b64encode(img)
        img = {
            "loss": delta,
            "rotation": bestrot,
            "shape": shape,
            "data": img
        }
        jsonCall.setData(img)

    def approachTarget(self, conn, jsonCall):
        keys = self.target.keys()
        step = jsonCall.getParam("step")
        modifiers = jsonCall.getParam("modifiers")
        values = self.lookupModifiers(modifiers)
        data = self.iterate(self.target, values, keys, step)
        jsonCall.setData(json.dumps(data))

    def landmarks(self, conn, jsonCall):
        lm = self.analyze()
        if lm:
            jsonCall.setData(json.dumps(lm))
        else:
            jsonCall.setError("ERROR: No face found")

    def maximizeWindow(self, conn, jsonCall):
        geo = G.app.mainwin.storeGeometry()
        print(geo)
        geo['x'] = 0
        geo['y'] = 0
        geo['width'] = 1920
        geo['height'] = 1080
        G.app.mainwin.restoreGeometry(geo)
        print(G.app.mainwin.storeGeometry())
        jsonCall.setData("OK")

    def resetCamera(self, conn, jsonCall):
        G.app.selectedHuman.setPosition([0.0, 0.0, 0.0])
        G.app.selectedHuman.setRotation([0.0, 0.0, 0.0])
        G.app.modelCamera.setPosition([0.0, 0.9, 0.0])
        G.app.modelCamera.setRotation([0.0, 0.0, 0.0])
        G.app.modelCamera.zoomFactor = 7
        jsonCall.setData("OK")

    def snapshot(self, conn, jsonCall):
        G.app.redraw()
        img = mh.grabScreen(0, 0, G.windowWidth, G.windowHeight)
        img = np.asarray(img.data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shape = img.shape
        img = img.reshape(-1)
        img = base64.b64encode(img)
        img = {
            "shape": shape,
            "data": img
        }
        # img = json.dumps(img)
        jsonCall.setData(img)

    def setTarget(self, conn, jsonCall):
        img = jsonCall.getParam('data')
        img = bytearray(img)
        img = img.decode()

        img = base64.b64decode(img)
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, -1)

        model = self.grabImage()
        if img.shape != model.shape:
            th, tw, tc = img.shape
            mh, mw, mc = model.shape

            if tc != mc:
                raise Exception("Image formats incompatible")

            maxh = max(th, mh)
            maxw = max(tw, mw)
            blank_image = np.zeros((maxh, maxw, 3), np.uint8)

            toffh = int((maxh - th)/2)
            toffw = int((maxw - tw)/2)
            blank_image[toffh:toffh + th, toffw:toffw + tw] = img

            moffh = int((maxh - mh)/2)
            moffw = int((maxw - mw)/2)
            img = blank_image[moffh:moffh + mh, moffw:moffw + mw]

        self.target = None
        self.target_image = img
        self.target_rotation = 0.0

        self.target = self.getLandmarks(img)

        shape = img.shape
        img = img.reshape(-1)
        img = base64.b64encode(img)
        img = {
            "shape": shape,
            "data": img
        }
        jsonCall.setData(img)

    def getModifierValue(self, conn, jsonCall):
        modifierName = jsonCall.getParam("modifier")
        if modifierName == 'all':
            list = self.api.modifiers.getAvailableModifierNames()
            jsonCall.setData(self.lookupModifiers(list))
            return
        else:
            try:
                val = self.lookupModifier(modifierName)
                d = {modifierName: val}
                jsonCall.setData(d)
                return
            except:
                print("Unknown modifier: " + modifierName)

    def getAvailableModifierNames(self,conn,jsonCall):
        jsonCall.data = self.api.modifiers.getAvailableModifierNames()

    def getAppliedTargets(self,conn,jsonCall):
        jsonCall.data = self.api.modifiers.getAppliedTargets()

    def applyModifier(self,conn,jsonCall):
        modifierName = jsonCall.getParam("modifier")
        if modifierName == 'all':
            values = jsonCall.getParam("power")
            for modifierName in values:
                try:
                    power = values[modifierName]
                    self.setModifier(modifierName, power, False)
                except:
                    print("Unknown modifier: "+modifierName)
            self.api.modifiers._threadSafeApplyAllTargets()
            jsonCall.setData("OK")
            return
        else:
            try:
                power = float(jsonCall.getParam("power"))
            except:
                jsonCall.setError("No such modifier")
                return

            self.setModifier(modifierName, power, True)
            jsonCall.setData("OK")

    def setModifier(self, modifierName, val, apply=False):
        if modifierName.startswith('camera/'):
            cmd = modifierName[7:]
            if cmd == 'zoom':
                G.app.modelCamera.zoomFactor = val
                G.app.redraw()
            elif (cmd.startswith('rot_')):
                axis = -1
                if cmd.endswith('x'):
                    axis = 0
                elif cmd.endswith('y'):
                    axis = 1
                elif cmd.endswith('z'):
                    axis = 2
                # G.app.rotateCamera(axis, power * 180)
                if axis != -1:
                    c = G.app.selectedHuman.getRotation()
                    c[axis] = val * 180
                    G.app.selectedHuman.setRotation(c)
            elif (cmd.startswith('trans_')):
                axis = -1
                if cmd.endswith('x'):
                    axis = 0
                elif cmd.endswith('y'):
                    axis = 1
                elif cmd.endswith('z'):
                    axis = 2
                if axis != -1:
                    c = G.app.selectedHuman.getPosition()
                    c[axis] = val
                    G.app.selectedHuman.setPosition(c)
        else:
            modifier = self.api.internals.getHuman().getModifier(modifierName)
            modifier.setValue(val)
        if apply:
            self.api.modifiers._threadSafeApplyAllTargets()
            # self.human.applyAllTargets()
            # G.app.redraw()

    def getLandmarks(self, imgRGB):
        NUM_FACE = 1
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
        results = faceMesh.process(imgRGB)
        faceMesh.close()
        data = {}
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                for id, lm in enumerate(faceLms.landmark):
                    data['X_' + str(id)] = lm.x * 1000
                    data['Y_' + str(id)] = lm.y * 1000
                    data['Z_' + str(id)] = lm.z * 1000
            return data
        else:
            return None

    def lookupModifiers(self, list):
        values = {}
        for modifierName in list:
            try:
                values[modifierName] = self.lookupModifier(modifierName)
            except:
                print("Unknown modifier: " + modifierName)
        return values

    def lookupModifier(self, modifierName):
        val = None
        if modifierName.startswith('camera/'):
            cmd = modifierName[7:]
            if cmd == 'zoom':
                val = G.app.modelCamera.zoomFactor
            elif cmd.startswith('rot_'):
                axis = -1
                if cmd.endswith('x'):
                    axis = 0
                elif cmd.endswith('y'):
                    axis = 1
                elif cmd.endswith('z'):
                    axis = 2
                if axis != -1:
                    c = G.app.selectedHuman.getRotation()
                    val = c[axis] / 180
            elif cmd.startswith('trans_'):
                axis = -1
                if cmd.endswith('x'):
                    axis = 0
                elif cmd.endswith('y'):
                    axis = 1
                elif cmd.endswith('z'):
                    axis = 2
                if axis != -1:
                    c = G.app.selectedHuman.getPosition()
                    val = c[axis]
        else:
            modifier = self.api.internals.getHuman().getModifier(modifierName)
            val = modifier.getValue()
        return val

    def calcdelta(self, d1, d2, keys):
        delta = 0
        for key in keys:
            d = float(d2[key]) - float(d1[key])
            delta += (d * d)
        return delta

    def grabImage(self):
        G.app.redraw()
        img = mh.grabScreen(0, 0, G.windowWidth, G.windowHeight)
        img = img.data
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return imgRGB

    def analyze(self):
        imgRGB = self.grabImage()
        return self.getLandmarks(imgRGB)

    def iterate(self, target, values, keys, step):
        count = 0
        for modifier in values:
            astep = step
            if modifier.startswith("camera/rot_"):
                astep = step/10

            v = values[modifier]
            self.setModifier(modifier, v, True)
            data = self.analyze()
            delta = self.calcdelta(target, data, keys)
            nuv = v + astep
            self.setModifier(modifier, nuv, True)
            nudata = self.analyze()
            nudelta = self.calcdelta(target, nudata, keys)
            if nudelta >= delta:
                nuv = v - astep
                self.setModifier(modifier, nuv, True)
                nudata = self.analyze()
                nudelta = self.calcdelta(target, nudata, keys)
                if nudelta >= delta:
                    nuv = v
            if v != nuv:
                count += 1
                print(modifier+': '+str(nudelta)+' / '+str(nuv))
            # else:
            #     done[modifier] = True
            self.setModifier(modifier, nuv, True)
            values[modifier] = nuv
        data = self.analyze()
        delta = self.calcdelta(target, data, keys)
        print("Value: "+str(delta))
        print("Count: "+str(count))
        d = {
            "count": count,
            "data": data,
            "loss": delta
        }
        return d

    def rotateImg(self, img, rotation, x, y):
        ih, iw, ic = img.shape
        rx2 = (iw / 2) - (iw * x / 1000)
        ry2 = (ih / 2) - (ih * y / 1000)
        img = imutils.translate(img, rx2, ry2)
        img = imutils.rotate(img, angle=rotation)
        img = imutils.translate(img, 0 - rx2, 0 - ry2)
        return img
