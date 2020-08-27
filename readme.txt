所有代码路径：jiachen@9.73.128.103:/advGAN
一、文件夹说明

	1、cycleGAN/models
		攻击模型的代码
	
	2、cycleGAN/results
		测试过程中生成的对抗样本结果
	
	3、cycleGAN/checkpoints
		训练过程中生成的攻击模型
	
	5、cycleGAN/run.sh
		运行攻击代码的指令，包括训练和测试两步，可以参照此格式进行修改

	4、0_final_results
		所有用到的最终结果，其中：
		img_spe 为img_spe模式的攻击结果图
		img_agn 为img_agn模式的攻击结果图
		compare 为两者对比的结果
		表格文件是human study的原始数据
	
	6、trainImages 
		训练和测试的数据集

	7、evaluate
		单张图片的测试代码 	
	
	8、writing
		文章书写中用到的图片

	9、photos_ones
		img_spe 模式的手机拍照图片

	10、photos_mix
		img_agn 模式的手机拍照图片

	11、evaluate_ones
		img_spe 模式的攻击结果及其评价代码

	12、evaluate_mix
		img_agn 模式的攻击结果及其评价代码


二、配置环境
	1、安装conda环境	
		conda create -n pytorch_env python=3.7  #pytorch是创造的环境名
		source activate pytorch_env   #进入环境（运行之前需要进入）

	2、cycleGAN所需要的环境：用pip install安装dominate 和 visdom 
		
	3、安装其他乱七八糟的支持库
	scikit-learn等，具体可以等到报错的时候再用pip安装
	
	