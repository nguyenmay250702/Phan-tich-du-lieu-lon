package map_reduce;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;




public class Main extends Configured implements Tool {
	
	public static PointWritable[] initRandomCentroids(int kClusters, int nLineOfInputFile, String inputFilePath,
			Configuration conf) throws IOException {
		System.out.println("Initializing random " + kClusters + " centroids...");
		PointWritable[] points = new PointWritable[kClusters];

		List<Integer> lstLinePos = new ArrayList<Integer>();
		Random random = new Random();
		int pos;
		while (lstLinePos.size() < kClusters) {
			pos = random.nextInt(nLineOfInputFile);
			if (!lstLinePos.contains(pos)) {
				lstLinePos.add(pos);
			}
		}
		Collections.sort(lstLinePos);

		FileSystem hdfs = FileSystem.get(conf);
		FSDataInputStream in = hdfs.open(new Path(inputFilePath));

		BufferedReader br = new BufferedReader(new InputStreamReader(in));

		int row = 0;
		int i = 0;
		while (i < lstLinePos.size()) {
			pos = lstLinePos.get(i);
			String point = br.readLine();
			if (row == pos) {
				points[i] = new PointWritable(point.split(","));
				i++;
			}
			row++;
		}
		br.close();
		return points;

	}

	public static void saveCentroidsForShared(Configuration conf, PointWritable[] points) {
		for (int i = 0; i < points.length; i++) {
			String centroidName = "C" + i;
			conf.unset(centroidName);
			conf.set(centroidName, points[i].toString());
		}
	}

	public static PointWritable[] readCentroidsFromReducerOutput(Configuration conf, int kClusters, String folderOutputPath) throws IOException, FileNotFoundException {
		PointWritable[] points = new PointWritable[kClusters];
		FileSystem hdfs = FileSystem.get(conf);
		FileStatus[] status = hdfs.listStatus(new Path(folderOutputPath));

		for (int i = 0; i < status.length; i++) {

			if (!status[i].getPath().toString().endsWith("_SUCCESS")) {
				Path outFilePath = status[i].getPath();
				System.out.println("read " + outFilePath.toString());
				BufferedReader br = new BufferedReader(new InputStreamReader(hdfs.open(outFilePath)));
				String line = null;// br.readLine();
				while ((line = br.readLine()) != null) {
					System.out.println(line);

					String[] strCentroidInfo = line.split("\t"); // Split line in K,V
					int centroidId = Integer.parseInt(strCentroidInfo[0]);
					String[] attrPoint = strCentroidInfo[1].split(",");
					points[centroidId] = new PointWritable(attrPoint);
				}
				br.close();
			}
		}

		hdfs.delete(new Path(folderOutputPath), true);

		return points;
	}

	private static boolean checkStopKMean(PointWritable[] oldCentroids, PointWritable[] newCentroids, float threshold) {
		boolean needStop = true;

		System.out.println("Check for stop K-Means if distance <= " + threshold);
		for (int i = 0; i < oldCentroids.length; i++) {

			double dist = oldCentroids[i].calcDistance(newCentroids[i]);
			System.out.println("distance centroid[" + i + "] changed: " + dist + " (threshold:" + threshold + ")");
			needStop = dist <= threshold;
			// chỉ cần 1 tâm < ngưỡng thì return false
			if (!needStop) {
				return false;
			}
		}
		return true;
	}

	private static void writeFinalResult(Configuration conf, PointWritable[] centroidsFound, String outputFilePath,
			PointWritable[] centroidsInit) throws IOException {
		FileSystem hdfs = FileSystem.get(conf);
		FSDataOutputStream dos = hdfs.create(new Path(outputFilePath), true);
		BufferedWriter br = new BufferedWriter(new OutputStreamWriter(dos));

		for (int i = 0; i < centroidsFound.length; i++) {
			br.write(centroidsFound[i].toString());
			br.newLine();
			System.out.println("Centroid[" + i + "]:  (" + centroidsFound[i] + ")  init: (" + centroidsInit[i] + ")");
		}

		br.close();
		hdfs.close();
	}

	public static PointWritable[] copyCentroids(PointWritable[] points) {
		PointWritable[] savedPoints = new PointWritable[points.length];
		for (int i = 0; i < savedPoints.length; i++) {
			savedPoints[i] = PointWritable.copy(points[i]);
		}
		return savedPoints;
	}

	public static int MAX_LOOP = 50;

	public static void printCentroids(PointWritable[] points, String name) {
		System.out.println("=> CURRENT CENTROIDS:");
		for (int i = 0; i < points.length; i++)
			System.out.println("centroids(" + name + ")[" + i + "]=> :" + points[i]);
		System.out.println("----------------------------------");
	}
	
	private static void addValue(Map<String, List<String>> map, String key, String value) {
        if (!map.containsKey(key)) {
            map.put(key, new ArrayList<>());
        }
        map.get(key).add(value);
    }
	
	//in ket qua dau ra
//	public void KetQua(Configuration conf, PointWritable[] centroidsFound, String outputFilePath) throws IOException {
//
//		Map<String, List<String>> dataMap = new HashMap<>();
//		FileSystem hdfs = FileSystem.get(conf);
//
//		  // Đọc dữ liệu từ HDFS
//		  Path hdfsPath = new Path("/k-input/data-kmeans.txt");
//		  try (FSDataInputStream inputStream = hdfs.open(hdfsPath);
//		       BufferedReader br_reader = new BufferedReader(new InputStreamReader(inputStream))) {
//			  
//			String line;
//			while ((line = br_reader.readLine()) != null) {
//			  String[] A = line.split(",");
//			  PointWritable P = new PointWritable(A);
//			  PointWritable C = new PointWritable();
//			  double minDistance = Double.MAX_VALUE;
//			  for (int i = 0; i < centroidsFound.length; i++) {
//			    double distance = P.calcDistance(centroidsFound[i]);
//			    if (distance < minDistance) {
//			      minDistance = distance;
//			      C = centroidsFound[i];
//			    }
//			  }
//			  addValue(dataMap, C.toString(), P.toString());
//			}
//		  }
//
//		  // Ghi dữ liệu vào HDFS
//		  try (FSDataOutputStream dos = hdfs.create(new Path(outputFilePath), true);
//		       BufferedWriter br_writer = new BufferedWriter(new OutputStreamWriter(dos))) {
//		    int i = 0;
//		    for (Map.Entry<String, List<String>> entry : dataMap.entrySet()) {
//		    	i++;
//		        br_writer.write("center" + i + ": ("+entry.getKey() + ")\npoints: " + entry.getValue().toString().replace(", ", ");  (").replace("[", "(").replace("]", ")") + "\n");
//			    br_writer.newLine();
//		    }
//		  }
//
//		  hdfs.close();
//		}

	//tải file kết quả về
	public void download_result(Configuration conf) {
		String hdfsFilePath = "/BTTH04/k-output/result.txt";
        String localDestinationPath = "D:\\phan_tich_du_lieu_lon\\TH\\BTTH04\\result.txt";
        try {
            FileSystem fs = FileSystem.get(conf);
            Path hdfsPath = new Path(hdfsFilePath);
            Path localPath = new Path(localDestinationPath);

            fs.copyToLocalFile(false, hdfsPath, localPath);
            System.out.println("File downloaded result successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
	}
	
	public int run(String[] args) throws Exception {

		Configuration conf = getConf();
		String inputFilePath = conf.get("in", null);
		String outputFolderPath = conf.get("out", null);
		String outputFileName = conf.get("result", "ClusterCenter.txt");

		int nClusters = conf.getInt("k", 3);
		float thresholdStop = conf.getFloat("thresh", 0.001f);
		int numLineOfInputFile = conf.getInt("lines", 0);
		MAX_LOOP = conf.getInt("maxloop", 50);
		int nReduceTask = conf.getInt("NumReduceTask", 1);
		if (inputFilePath == null || outputFolderPath == null || numLineOfInputFile == 0) {
			System.err.printf(
					"Usage: %s -Din <input file name> -Dlines <number of lines in input file> -Dout <Folder ouput> -Dresult <output file result> -Dk <number of clusters> -Dthresh <Threshold>\n",
					getClass().getSimpleName());
			ToolRunner.printGenericCommandUsage(System.err);
			return -1;
		}
		System.out.println("---------------INPUT PARAMETERS---------------");
		System.out.println("inputFilePath:" + inputFilePath);
		System.out.println("outputFolderPath:" + outputFolderPath);
		System.out.println("outputFileName:" + outputFileName);
		System.out.println("maxloop:" + MAX_LOOP);
		System.out.println("numLineOfInputFile:" + numLineOfInputFile);
		System.out.println("nClusters:" + nClusters);
		System.out.println("threshold:" + thresholdStop);
		System.out.println("NumReduceTask:" + nReduceTask);

		System.out.println("--------------- STATR ---------------");
		PointWritable[] oldCentroidPoints = initRandomCentroids(nClusters, numLineOfInputFile, inputFilePath, conf);
		PointWritable[] centroidsInit = copyCentroids(oldCentroidPoints);
		printCentroids(oldCentroidPoints, "init");
		saveCentroidsForShared(conf, oldCentroidPoints);
		int nLoop = 0;
		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path("/BTTH04/map_reduce"), true); 
		fs.delete(new Path("/BTTH04/k-output"), true); 

		PointWritable[] newCentroidPoints = null;
		long t1 = (new Date()).getTime();
		while (true) {
			nLoop++;
			if (nLoop == MAX_LOOP) {
				break;
			}
			int lanlap = nLoop;
			conf.set("nLoop", String.valueOf(String.valueOf(lanlap)));
			
			Job job = new Job(conf, "K-Mean");// Job thực hiện deepCopy conf
			
			job.setJarByClass(Main.class);
			job.setMapperClass(KMapper.class);
			job.setCombinerClass(KCombiner.class);
			job.setReducerClass(KReducer.class);
			job.setMapOutputKeyClass(LongWritable.class);
			job.setMapOutputValueClass(PointWritable.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);
			
			FileInputFormat.addInputPath(job, new Path(inputFilePath));

			FileOutputFormat.setOutputPath(job, new Path(outputFolderPath));
			job.setOutputFormatClass(TextOutputFormat.class);

			job.setNumReduceTasks(nReduceTask);

			boolean ret = job.waitForCompletion(true);
			if (!ret) {
				return -1;
			}

			newCentroidPoints = readCentroidsFromReducerOutput(conf, nClusters, outputFolderPath);
			printCentroids(newCentroidPoints, "new");
			boolean needStop = checkStopKMean(newCentroidPoints, oldCentroidPoints, thresholdStop);

			oldCentroidPoints = copyCentroids(newCentroidPoints);

			if (needStop) {
				break;
			} else {
				saveCentroidsForShared(conf, newCentroidPoints);
			}

		}
		if (newCentroidPoints != null) {
			System.out.println("------------------- FINAL RESULT -------------------");
			writeFinalResult(conf, newCentroidPoints, outputFolderPath + "/" + outputFileName, centroidsInit);
//			KetQua(conf, newCentroidPoints, outputFolderPath+"/kq_phancum.txt");
			download_result(conf);
		}
		System.out.println("----------------------------------------------");
		System.out.println("K-MEANS CLUSTERING FINISHED!");
		System.out.println("Loop:" + nLoop);
		System.out.println("Time:" + ((new Date()).getTime() - t1) + "ms");

		return 1;
	}

	public static void main(String[] args) throws Exception {
		int exitCode = ToolRunner.run(new Main(), args);
		System.exit(exitCode);
	}
}
