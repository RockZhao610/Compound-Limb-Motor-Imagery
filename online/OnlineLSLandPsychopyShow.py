#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.1.2),
    on 五月 05, 2024, at 20:25
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code
from pylsl import StreamInlet, resolve_stream
# Run 'Begin Routine' code from code_2
# Pylsl发送端
from pylsl import StreamInfo, StreamOutlet,StreamInlet, resolve_stream
from scipy.io import savemat
import serial
import time
from pylsl import StreamInlet, resolve_stream, local_clock

def send_marker(data, marker_outlet):
    # 获取当前时间作为时间戳
    marker_timestamp = time.time()
    
    # 打印发送数据的信息
    print("***************************************")
    print('Now sending data: ', data)
    print("***************************************")

    # 将数据和时间戳一起发送
    marker_outlet.push_sample([data], marker_timestamp)


           

markerOutPut_info = StreamInfo('StartpredictionStream', 'Markers', 1, 0, 'string', 'myuidw43539')
marker_outlet = StreamOutlet(markerOutPut_info)

#解析数据流
streams = resolve_stream('type', 'Markers3')

#创建流的对象
inlet = StreamInlet(streams[0])

keys = event.getKeys()
if 'space' in keys:  # 假设我们使用空格键作为继续的标记
    continueRoutine = False  # 结束当前Routine
elif 'escape' in keys:  # 允许使用esc键退出
    core.quit()  # 完全退出实验




# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2023.1.2'
expName = 'OnlineMI'  # from the Builder filename that created this script
expInfo = {
    'participant': 'RUI ',
    'session': '001',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='F:\\offline\\offlineMI55.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[1280, 768], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0.0039, 0.0039, 0.0039], colorSpace='rgb',
    backgroundImage='', backgroundFit='none',
    blendMode='avg', useFBO=False, 
    units='height')
win.mouseVisible = True
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "WelcomeScreen" ---
textWelcomeMessage = visual.TextStim(win=win, name='textWelcomeMessage',
    text='Welcome to the experiment\n',
    font='Times New Roman',
    pos=(0, 0), height=0.1, wrapWidth=10.0, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "startblock" ---
textstartblock = visual.TextStim(win=win, name='textstartblock',
    text='Blcok Start',
    font='Times New Roman',
    pos=(0, 0), height=0.2, wrapWidth=10.0, ori=0.0, 
    color=[0.3569, 0.6941, 0.8039], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "cross" ---
baseline = visual.ShapeStim(
    win=win, name='baseline', vertices='cross',
    size=(0.5, 0.5),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.0039, 0.0039, 0.0039], fillColor=[-1.0000, -1.0000, -1.0000],
    opacity=None, depth=0.0, interpolate=True)

# --- Initialize components for Routine "mi" ---
moviePresentation = visual.MovieStim(
    win, name='moviePresentation',
    filename=None, movieLib='ffpyplayer',
    loop=False, volume=1.0, noAudio=False,
    pos=(0, 0), size=(1.5, 1), units=win.units,
    ori=0.0, anchor='center',opacity=None, contrast=1.0,
    depth=0
)

# --- Initialize components for Routine "waitInput" ---

# --- Initialize components for Routine "prepare" ---
text = visual.TextStim(win=win, name='text',
    text='exoskeleton moving',
    font='Open Sans',
    pos=(0, 0), height=0.2, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "endblock" ---
textendblock = visual.TextStim(win=win, name='textendblock',
    text='Block End',
    font='Times New Roman',
    pos=(0, 0), height=0.2, wrapWidth=10.0, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "GoodbyeScreen" ---
textGoodbyeScreen = visual.TextStim(win=win, name='textGoodbyeScreen',
    text='Thanks for your participation',
    font='Times New Roman',
    pos=(0, 0), height=0.1, wrapWidth=10.0, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "WelcomeScreen" ---
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
WelcomeScreenComponents = [textWelcomeMessage]
for thisComponent in WelcomeScreenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "WelcomeScreen" ---
routineForceEnded = not continueRoutine
while continueRoutine and routineTimer.getTime() < 2.0:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *textWelcomeMessage* updates
    
    # if textWelcomeMessage is starting this frame...
    if textWelcomeMessage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        textWelcomeMessage.frameNStart = frameN  # exact frame index
        textWelcomeMessage.tStart = t  # local t and not account for scr refresh
        textWelcomeMessage.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(textWelcomeMessage, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'textWelcomeMessage.started')
        # update status
        textWelcomeMessage.status = STARTED
        textWelcomeMessage.setAutoDraw(True)
    
    # if textWelcomeMessage is active this frame...
    if textWelcomeMessage.status == STARTED:
        # update params
        pass
    
    # if textWelcomeMessage is stopping this frame...
    if textWelcomeMessage.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > textWelcomeMessage.tStartRefresh + 2.0-frameTolerance:
            # keep track of stop time/frame for later
            textWelcomeMessage.tStop = t  # not accounting for scr refresh
            textWelcomeMessage.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textWelcomeMessage.stopped')
            # update status
            textWelcomeMessage.status = FINISHED
            textWelcomeMessage.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        if eyetracker:
            eyetracker.setConnectionState(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in WelcomeScreenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "WelcomeScreen" ---
for thisComponent in WelcomeScreenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
if routineForceEnded:
    routineTimer.reset()
else:
    routineTimer.addTime(-2.000000)

# set up handler to look after randomisation of conditions etc
blocksloop = data.TrialHandler(nReps=6.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('cond_file.xlsx'),
    seed=1, name='blocksloop')
thisExp.addLoop(blocksloop)  # add the loop to the experiment
thisBlocksloop = blocksloop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlocksloop.rgb)
if thisBlocksloop != None:
    for paramName in thisBlocksloop:
        exec('{} = thisBlocksloop[paramName]'.format(paramName))

for thisBlocksloop in blocksloop:
    currentLoop = blocksloop
    # abbreviate parameter names if possible (e.g. rgb = thisBlocksloop.rgb)
    if thisBlocksloop != None:
        for paramName in thisBlocksloop:
            exec('{} = thisBlocksloop[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "startblock" ---
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    startblockComponents = [textstartblock]
    for thisComponent in startblockComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "startblock" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textstartblock* updates
        
        # if textstartblock is starting this frame...
        if textstartblock.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textstartblock.frameNStart = frameN  # exact frame index
            textstartblock.tStart = t  # local t and not account for scr refresh
            textstartblock.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textstartblock, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textstartblock.started')
            # update status
            textstartblock.status = STARTED
            textstartblock.setAutoDraw(True)
        
        # if textstartblock is active this frame...
        if textstartblock.status == STARTED:
            # update params
            pass
        
        # if textstartblock is stopping this frame...
        if textstartblock.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textstartblock.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                textstartblock.tStop = t  # not accounting for scr refresh
                textstartblock.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textstartblock.stopped')
                # update status
                textstartblock.status = FINISHED
                textstartblock.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
            if eyetracker:
                eyetracker.setConnectionState(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in startblockComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "startblock" ---
    for thisComponent in startblockComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.00000)
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=10.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(thisBlocksloop.cond_file),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    for thisTrial in trials:
        currentLoop = trials
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "cross" ---
        continueRoutine = True
        # update component parameters for each repeat
     
        # TODO send the marker
        send_marker(thisTrial.marker, marker_outlet)
        # keep track of which components have finished
        crossComponents = [baseline]
        for thisComponent in crossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "cross" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *baseline* updates
            
            # if baseline is starting this frame...
            if baseline.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                baseline.frameNStart = frameN  # exact frame index
                baseline.tStart = t  # local t and not account for scr refresh
                baseline.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(baseline, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'baseline.started')
                # update status
                baseline.status = STARTED
                baseline.setAutoDraw(True)
            
            # if baseline is active this frame...
            if baseline.status == STARTED:
                # update params
                pass
            
            # if baseline is stopping this frame...
            if baseline.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > baseline.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    baseline.tStop = t  # not accounting for scr refresh
                    baseline.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'baseline.stopped')
                    # update status
                    baseline.status = FINISHED
                    baseline.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
                if eyetracker:
                    eyetracker.setConnectionState(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in crossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "cross" ---
        for thisComponent in crossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "mi" ---
        continueRoutine = True
        # update component parameters for each repeat
        moviePresentation.setMovie(thisTrial.action_movies)
        # keep track of which components have finished
        miComponents = [moviePresentation]
        for thisComponent in miComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mi" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *moviePresentation* updates
            
            # if moviePresentation is starting this frame...
            if moviePresentation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moviePresentation.frameNStart = frameN  # exact frame index
                moviePresentation.tStart = t  # local t and not account for scr refresh
                moviePresentation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moviePresentation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moviePresentation.started')
                # update status
                moviePresentation.status = STARTED
                moviePresentation.setAutoDraw(True)
                moviePresentation.play()
            
            # if moviePresentation is stopping this frame...
            if moviePresentation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moviePresentation.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    moviePresentation.tStop = t  # not accounting for scr refresh
                    moviePresentation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moviePresentation.stopped')
                    # update status
                    moviePresentation.status = FINISHED
                    moviePresentation.setAutoDraw(False)
                    moviePresentation.stop()
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
                if eyetracker:
                    eyetracker.setConnectionState(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in miComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mi" ---
        for thisComponent in miComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        moviePresentation.stop()
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        
        # --- Prepare to start Routine "waitInput" ---
        continueRoutine = True
        # update component parameters for each repeat
        # keep track of which components have finished
        waitInputComponents = []
        for thisComponent in waitInputComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "waitInput" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
                if eyetracker:
                    eyetracker.setConnectionState(False)
            # 设置流解析，寻找特定类型的标记流
            # TODO
           # 尝试解析流
            
# 尝试解析流
                
            if streams:
                print("Stream found")
               
                start_time = local_clock()
                received_marker = False  # 初始化标记检测变量
            
                # 维持监听直到收到特定标记或超时
                while not received_marker and local_clock() - start_time < 5:  # 增加总监听时间至30秒
                    try:
                        sample, timestamp = inlet.pull_sample(timeout=3.0)  # 增加超时时间至5秒
                        if sample:
                            print("Sample received:", sample, "at", timestamp)
                            # 检查是否收到特定标记
                            if sample[0] == 'OK':
                                received_marker = True
                                print("Found and fit marker: OK")
                            else:
                                print("Found the marker but not fit:", sample[0])
                        else:
                            print("Waiting for sample...")
                    except Exception as e:
                        print("Error during sampling:", str(e))
            
                # 循环结束后根据是否收到标记进行处理
                if received_marker:
                    print("Marker OK received within the time limit.")
                else:
                    print("No suitable marker received within the time limit.")
            
            else:
                print("No streams found of type 'Markers3'")

            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
                if eyetracker:
                    eyetracker.setConnectionState(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in waitInputComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "waitInput" ---
        for thisComponent in waitInputComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "waitInput" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prepare" ---
        continueRoutine = True
        # update component parameters for each repeat
        # keep track of which components have finished
        prepareComponents = [text]
        for thisComponent in prepareComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prepare" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
                if eyetracker:
                    eyetracker.setConnectionState(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prepareComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prepare" ---
        for thisComponent in prepareComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        thisExp.nextEntry()
        
    # completed 10.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "endblock" ---
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    endblockComponents = [textendblock]
    for thisComponent in endblockComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "endblock" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 45.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textendblock* updates
        
        # if textendblock is starting this frame...
        if textendblock.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textendblock.frameNStart = frameN  # exact frame index
            textendblock.tStart = t  # local t and not account for scr refresh
            textendblock.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textendblock, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textendblock.started')
            # update status
            textendblock.status = STARTED
            textendblock.setAutoDraw(True)
        
        # if textendblock is active this frame...
        if textendblock.status == STARTED:
            # update params
            pass
        
        # if textendblock is stopping this frame...
        if textendblock.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textendblock.tStartRefresh + 45-frameTolerance:
                # keep track of stop time/frame for later
                textendblock.tStop = t  # not accounting for scr refresh
                textendblock.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textendblock.stopped')
                # update status
                textendblock.status = FINISHED
                textendblock.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
            if eyetracker:
                eyetracker.setConnectionState(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endblockComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "endblock" ---
    for thisComponent in endblockComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-45.000000)
# completed 6.0 repeats of 'blocksloop'


# --- Prepare to start Routine "GoodbyeScreen" ---
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
GoodbyeScreenComponents = [textGoodbyeScreen]
for thisComponent in GoodbyeScreenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "GoodbyeScreen" ---
routineForceEnded = not continueRoutine
while continueRoutine and routineTimer.getTime() < 2.0:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *textGoodbyeScreen* updates
    
    # if textGoodbyeScreen is starting this frame...
    if textGoodbyeScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        textGoodbyeScreen.frameNStart = frameN  # exact frame index
        textGoodbyeScreen.tStart = t  # local t and not account for scr refresh
        textGoodbyeScreen.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(textGoodbyeScreen, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'textGoodbyeScreen.started')
        # update status
        textGoodbyeScreen.status = STARTED
        textGoodbyeScreen.setAutoDraw(True)
    
    # if textGoodbyeScreen is active this frame...
    if textGoodbyeScreen.status == STARTED:
        # update params
        pass
    
    # if textGoodbyeScreen is stopping this frame...
    if textGoodbyeScreen.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > textGoodbyeScreen.tStartRefresh + 2-frameTolerance:
            # keep track of stop time/frame for later
            textGoodbyeScreen.tStop = t  # not accounting for scr refresh
            textGoodbyeScreen.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textGoodbyeScreen.stopped')
            # update status
            textGoodbyeScreen.status = FINISHED
            textGoodbyeScreen.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        if eyetracker:
            eyetracker.setConnectionState(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in GoodbyeScreenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "GoodbyeScreen" ---
for thisComponent in GoodbyeScreenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
if routineForceEnded:
    routineTimer.reset()
else:
    routineTimer.addTime(-2.000000)

# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
