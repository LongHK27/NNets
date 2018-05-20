using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.IO;

namespace TranscriptDetection
{
    public partial class Form1 : Form
    {
        public string[] files;
        public int index = 0;

        public Form1()
        {
            InitializeComponent();

            string path = Directory.GetCurrentDirectory().ToString() + "\\data\\";
            Parameters parameters = new Parameters();

            //Defining the default parameters
            parameters.layers_size = new int[] { 784, 100, 10 };
            parameters.nblayers = parameters.layers_size.Length;
            parameters.eta = 1;
            parameters.batchsize = 10;
            parameters.costfunction = Parameters.Costfunction.CROSSENTROPY;
            parameters.regularization = true;
            parameters.lambda = 3;
            parameters.stopafternbsteps = 20;
        }

        private void selectImage_Click(object sender, EventArgs e)
        {
            var dir = new FolderBrowserDialog();
            if (dir.ShowDialog() == DialogResult.OK)
            {
                files = System.IO.Directory.GetFiles(dir.SelectedPath);

                using (var source = new Image<Bgr, byte>(files[index]))
                {
                    TranscriptDetector.Detect(source);
                }
            }
        }

        private void next_Click(object sender, EventArgs e)
        {
            if (files.Length == 0) return;
            if (index == files.Length - 1) index = 0;
            else index++;
            using (var source = new Image<Bgr, byte>(files[index]))
            {
                TranscriptDetector.Detect(source);
            }
        }

        private void prev_Click(object sender, EventArgs e)
        {
            if (files.Length == 0) return;
            if (index == 0) index = files.Length - 1;
            else index--;
            using (var source = new Image<Bgr, byte>(files[index]))
            {
                TranscriptDetector.Detect(source);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {

            var file_list = System.IO.Directory.GetFiles(@"C:\xxx");
                
            for(var i = 0; i < file_list.Count<string>(); i++)
            {
                using (var source = new Image<Gray, byte>(file_list[i]))
                {
                    TranscriptDetector.DetectDigit(source);
                    CvInvoke.WaitKey(1);
                }
            }

        }

        private void button2_Click(object sender, EventArgs e)
        {

            

        }

        private void button3_Click(object sender, EventArgs e)
        {
            
        }


    }
}
