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

            NeuronNets.Init(path);
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

            NeuronNets.Training();

        }

        private void button3_Click(object sender, EventArgs e)
        {
            var dialog = new OpenFileDialog();
            if(dialog.ShowDialog() == DialogResult.OK)
            {
                using( var image = (new Image<Gray, byte>(dialog.FileName)).ThresholdBinaryInv(new Gray(180), new Gray(255)))
                {
                    CvInvoke.Resize(image, image, new Size(28, 28));

                    pictureBox1.Image = image.Bitmap;
                    pictureBox1.Refresh();

                    label3.Text = NeuronNets.Active(image).ToString();
                    label3.Refresh();
                }
            }
        }


    }
}
