# TensorPlace
Place wiv tensors

Right now you can train a model to reproduce audio on you own dataset. I tought pytorch also supports mp3 and such, but you can figure that out yourself.
Put you dataset in TensorPlace/DATA/WAV and press train. stdout shows progress. every so often it saves the model and produces a track. 
the first part of the track will be the input it got, then will be the output it gave, followed with auto-inference. once all sections are indestinguishable, it's done a good job.

hurdles:
- no cuda, microsoft directl-ml is very much alpha and unusable in this project, i tried my damndest but gpu memory usage was 4x cpu mode memory usage so what can you do. also cpu training is faster. Donate so can buy a h100.
- Definitely underestimated the required training time to become Jimi Hendrix.
- so, the input size is a balance between too long and too short. the shorter the worse performance in speed but "better" training. too long input size didn't allow for generalization in my lifetime. in terms of sequence length a file will generall be cut up in 1 min section. this is dependant on the sequence and input_size.
- Shoutout to GREG @ https://www.youtube.com/@greggggg for being a papa to my model.

  notes:
  - DELETE THE OLD CACHE IF YOU CHANGE INPUT SIZE FOR THE NETWORK. the dataset gets cached automagically because resampling takes a long time even with the fast resampler. the cached data is in an almost final form so you would have to reshape it to use a different notwork shape. the chache is save by samplerate, so changing that is also a good option then you can keep both networks and both sets in different rates without issue.
