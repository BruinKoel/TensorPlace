# TensorPlace
Place wiv tensors

Right now you can train a model to reproduce audio on you own dataset. I tought pytorch also supports mp3 and such, but you can figure that out yourself.
Put you dataset in TensorPlace/DATA/WAV and press train. stdout shows progress. every so often it saves the model and produces at track. 
the first part of the track will be the input it got, then will be the output it gave, or guessed followed with auto-inference. once all sections are indestinguishable, it's done a good job.

hurdles:
- no cuda, microsoft directl-ml is ver much alpha and unusable in this project, i tried mij damndest but gpu memory useage was 4x cpu mode memory usage so what can you do. also cpu training is faster .
- Definitely underestimated the requiered training time to become Jimi Hendrix.
- so, the input size is a balance between too long and too short. the shorter the worse performance in speed but "better" training. too long input size didn't allow for generalization in my lifetime.
- Shoutout to GREG @ https://www.youtube.com/@greggggg for being a papa to my model.

  notes:
  - yes
