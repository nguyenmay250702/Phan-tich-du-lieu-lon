package view;

import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

import map_reduce.Main;
import map_reduce.PointWritable;

import javax.swing.JLabel;
import javax.swing.JOptionPane;

import java.awt.Font;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JTextField;
import java.awt.GridLayout;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import javax.swing.border.LineBorder;
import java.awt.Color;
import javax.swing.border.TitledBorder;
import javax.swing.border.EtchedBorder;
import javax.swing.JButton;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.awt.event.ActionEvent;
import javax.swing.SwingConstants;
import java.awt.Window.Type;

public class Display extends JFrame {

	private JPanel contentPane;
	private JTextField textField_Ph;
	private JTextField textField_NhietDo;
	private JTextField textField_HuongVi;
	private JTextField textField_Mui;
	private JTextField textField_ChatBeo;
	private JTextField textField_DoTrong;
	private JTextField textField_Mau;
	private JLabel lbl_KQ;
	private JButton btn_PhanCum;

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					Display frame = new Display();
					frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}
	
	//kiểm tra giá trị có phải là số >=0 hay không
	private boolean isPositiveNumeric(String value) {
        try {
            double number = Double.parseDouble(value);
            return number >= 0;
        } catch (NumberFormatException e) {
            return false;
        }
    }
	
	//kiểm tra dữ liệu đầu vào
	public boolean check_data() {
		if(textField_Ph.getText() =="" || isPositiveNumeric(textField_Ph.getText()) == false) {
			textField_Ph.requestFocus(true);
			JOptionPane.showMessageDialog(null, "Giá trị 'Ph' phải là số >= 0", "Lỗi", JOptionPane.INFORMATION_MESSAGE);
			return false;
		}
		if(textField_NhietDo.getText() =="" || isPositiveNumeric(textField_NhietDo.getText()) == false) {
			textField_NhietDo.requestFocus(true);
			JOptionPane.showMessageDialog(null, "Giá trị 'Nhiệt Độ' phải là số >= 0", "Lỗi", JOptionPane.INFORMATION_MESSAGE);
			return false;
		}
		if(textField_HuongVi.getText() =="" || isPositiveNumeric(textField_HuongVi.getText()) == false) {
			textField_HuongVi.requestFocus(true);
			JOptionPane.showMessageDialog(null, "Giá trị 'Hương Vị' phải là số >= 0", "Lỗi", JOptionPane.INFORMATION_MESSAGE);
			return false;
		}
		if(textField_Mui.getText() =="" || isPositiveNumeric(textField_Mui.getText()) == false) {
			textField_Mui.requestFocus(true);
			JOptionPane.showMessageDialog(null, "Giá trị 'Mùi' phải là số >= 0", "Lỗi", JOptionPane.INFORMATION_MESSAGE);
			return false;
		}
		if(textField_ChatBeo.getText() =="" || isPositiveNumeric(textField_ChatBeo.getText()) == false) {
			textField_ChatBeo.requestFocus(true);
			JOptionPane.showMessageDialog(null, "Giá trị 'Chất Béo' phải là số >= 0", "Lỗi", JOptionPane.INFORMATION_MESSAGE);
			return false;
		}
		if(textField_DoTrong.getText() =="" || isPositiveNumeric(textField_DoTrong.getText()) == false) {
			textField_DoTrong.requestFocus(true);
			JOptionPane.showMessageDialog(null, "Giá trị 'Độ Trong' phải là số >= 0", "Lỗi", JOptionPane.INFORMATION_MESSAGE);
			return false;
		}
		if(textField_Mau.getText() =="" || isPositiveNumeric(textField_Mau.getText()) == false) {
			textField_Mau.requestFocus(true);
			JOptionPane.showMessageDialog(null, "Giá trị 'Màu' phải là số >= 0", "Lỗi", JOptionPane.INFORMATION_MESSAGE);
			return false;
		}		
		return true;
	}
	
	//phân cụm dựa trên dữ liệu nhập vào
	public void clustering() {
		//kiểm tra xem kết quả chứa tâm cụm đã được lưu về máy hay chưa
		File file = new File(Main.path_download_cluster_centers);
	    if (!file.exists()) {
			JOptionPane.showMessageDialog(null, "- Bạn cần chạy mô hình Kmean mapreduce trên hadoop trước.\n- Kiểm tra lại đường dẫn lưu file kết quả.", "Lỗi", JOptionPane.INFORMATION_MESSAGE);
	        return;
	    }
	    
	    //kiểm tra dữ liệu đầu vào + tính khoảng cách từ điểm đầu vào đến từng tâm cụm => lưu lại id tâm cụm có k/c nhỏ nhất đến điêm đầu vào
		if(check_data()==true) {
			PointWritable Point = new PointWritable(new String[] {textField_Ph.getText(), textField_NhietDo.getText(), textField_HuongVi.getText(), textField_Mui.getText(), textField_ChatBeo.getText(), textField_DoTrong.getText(), textField_Mau.getText()});
    		PointWritable Centroid = new PointWritable();
			
    		double minDistance = Double.MAX_VALUE;
    		int centroidIdNearest = 0;
    		int stt = 0;
	        try (BufferedReader br = new BufferedReader(new FileReader(Main.path_download_cluster_centers))) {
	            String line;
	            while ((line = br.readLine()) != null) {
	        		String[] arrValue = line.toString().split(",");
	        		Centroid.set(arrValue);
	        		double distance = Point.calcDistance(Centroid);
	        		
	    			if (distance < minDistance) {
	    				centroidIdNearest = stt;
	    				minDistance = distance;
	    			}
	    			stt++;
	            }
	            lbl_KQ.setText(String.valueOf(centroidIdNearest));
	            
	        } catch (Exception e) {
	            e.printStackTrace();
	        }
		}
	}
	

	/**
	 * Create the frame.
	 */
	public Display() {
		setTitle("Phần mềm phân cụm dữ liệu sữa");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 721, 567);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));

		setContentPane(contentPane);
		
		JLabel lblNewLabel = new JLabel("Phân cụm sữa sử dụng thuật toán k-mean");
		lblNewLabel.setForeground(new Color(29, 34, 129));
		lblNewLabel.setBounds(184, 10, 353, 21);
		lblNewLabel.setFont(new Font("Tahoma", Font.BOLD, 17));
		contentPane.setLayout(null);
		contentPane.add(lblNewLabel);
		
		JPanel panel = new JPanel();
		panel.setBorder(new TitledBorder(new EtchedBorder(EtchedBorder.LOWERED, new Color(255, 255, 255), new Color(160, 160, 160)), "Input", TitledBorder.LEFT, TitledBorder.TOP, null, new Color(0, 0, 0)));
		panel.setBounds(48, 78, 321, 380);
		contentPane.add(panel);
		GridBagLayout gbl_panel = new GridBagLayout();
		gbl_panel.columnWidths = new int[] {30, 0, 150, 20};
		gbl_panel.rowHeights = new int[] {40, 40, 40, 40, 40, 40, 40, 30, 30, 10};
		gbl_panel.columnWeights = new double[]{1.0, 1.0, 1.0, Double.MIN_VALUE};
		gbl_panel.rowWeights = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		panel.setLayout(gbl_panel);
		
		btn_PhanCum = new JButton("Phân cụm");
		btn_PhanCum.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				clustering();
			}
		});
		GridBagConstraints gbc_btn_PhanCum = new GridBagConstraints();
		gbc_btn_PhanCum.insets = new Insets(0, 0, 5, 0);
		gbc_btn_PhanCum.anchor = GridBagConstraints.WEST;
		gbc_btn_PhanCum.gridx = 2;
		gbc_btn_PhanCum.gridy = 8;
		panel.add(btn_PhanCum, gbc_btn_PhanCum);
		
		JLabel lblNewLabel_1 = new JLabel("PH");
		lblNewLabel_1.setFont(new Font("Tahoma", Font.PLAIN, 13));
		GridBagConstraints gbc_lblNewLabel_1 = new GridBagConstraints();
		gbc_lblNewLabel_1.anchor = GridBagConstraints.EAST;
		gbc_lblNewLabel_1.fill = GridBagConstraints.VERTICAL;
		gbc_lblNewLabel_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_1.gridx = 0;
		gbc_lblNewLabel_1.gridy = 0;
		panel.add(lblNewLabel_1, gbc_lblNewLabel_1);
		
		textField_Ph = new JTextField();
		textField_Ph.setFont(new Font("Tahoma", Font.PLAIN, 13));
		textField_Ph.setColumns(10);
		GridBagConstraints gbc_textField_Ph = new GridBagConstraints();
		gbc_textField_Ph.insets = new Insets(0, 0, 5, 0);
		gbc_textField_Ph.fill = GridBagConstraints.HORIZONTAL;
		gbc_textField_Ph.gridx = 2;
		gbc_textField_Ph.gridy = 0;
		panel.add(textField_Ph, gbc_textField_Ph);
		
		JLabel lblNewLabel_1_1 = new JLabel("Nhiệt độ");
		lblNewLabel_1_1.setFont(new Font("Tahoma", Font.PLAIN, 13));
		GridBagConstraints gbc_lblNewLabel_1_1 = new GridBagConstraints();
		gbc_lblNewLabel_1_1.anchor = GridBagConstraints.EAST;
		gbc_lblNewLabel_1_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_1_1.gridx = 0;
		gbc_lblNewLabel_1_1.gridy = 1;
		panel.add(lblNewLabel_1_1, gbc_lblNewLabel_1_1);
		
		textField_NhietDo = new JTextField();
		textField_NhietDo.setFont(new Font("Tahoma", Font.PLAIN, 13));
		textField_NhietDo.setColumns(10);
		GridBagConstraints gbc_textField_NhietDo = new GridBagConstraints();
		gbc_textField_NhietDo.insets = new Insets(0, 0, 5, 0);
		gbc_textField_NhietDo.fill = GridBagConstraints.HORIZONTAL;
		gbc_textField_NhietDo.gridx = 2;
		gbc_textField_NhietDo.gridy = 1;
		panel.add(textField_NhietDo, gbc_textField_NhietDo);
		
		JLabel lblNewLabel_1_1_1 = new JLabel("Hương vị");
		lblNewLabel_1_1_1.setFont(new Font("Tahoma", Font.PLAIN, 13));
		GridBagConstraints gbc_lblNewLabel_1_1_1 = new GridBagConstraints();
		gbc_lblNewLabel_1_1_1.anchor = GridBagConstraints.EAST;
		gbc_lblNewLabel_1_1_1.fill = GridBagConstraints.VERTICAL;
		gbc_lblNewLabel_1_1_1.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_1_1_1.gridx = 0;
		gbc_lblNewLabel_1_1_1.gridy = 2;
		panel.add(lblNewLabel_1_1_1, gbc_lblNewLabel_1_1_1);
		
		textField_HuongVi = new JTextField();
		textField_HuongVi.setFont(new Font("Tahoma", Font.PLAIN, 13));
		textField_HuongVi.setColumns(10);
		GridBagConstraints gbc_textField_HuongVi = new GridBagConstraints();
		gbc_textField_HuongVi.insets = new Insets(0, 0, 5, 0);
		gbc_textField_HuongVi.fill = GridBagConstraints.HORIZONTAL;
		gbc_textField_HuongVi.gridx = 2;
		gbc_textField_HuongVi.gridy = 2;
		panel.add(textField_HuongVi, gbc_textField_HuongVi);
		
		JLabel lblNewLabel_1_1_2 = new JLabel("Mùi");
		lblNewLabel_1_1_2.setFont(new Font("Tahoma", Font.PLAIN, 13));
		GridBagConstraints gbc_lblNewLabel_1_1_2 = new GridBagConstraints();
		gbc_lblNewLabel_1_1_2.anchor = GridBagConstraints.EAST;
		gbc_lblNewLabel_1_1_2.fill = GridBagConstraints.VERTICAL;
		gbc_lblNewLabel_1_1_2.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_1_1_2.gridx = 0;
		gbc_lblNewLabel_1_1_2.gridy = 3;
		panel.add(lblNewLabel_1_1_2, gbc_lblNewLabel_1_1_2);
		
		textField_Mui = new JTextField();
		textField_Mui.setFont(new Font("Tahoma", Font.PLAIN, 13));
		textField_Mui.setColumns(10);
		GridBagConstraints gbc_textField_Mui = new GridBagConstraints();
		gbc_textField_Mui.insets = new Insets(0, 0, 5, 0);
		gbc_textField_Mui.fill = GridBagConstraints.HORIZONTAL;
		gbc_textField_Mui.gridx = 2;
		gbc_textField_Mui.gridy = 3;
		panel.add(textField_Mui, gbc_textField_Mui);
		
		JLabel lblNewLabel_1_1_3 = new JLabel("Chất béo");
		lblNewLabel_1_1_3.setFont(new Font("Tahoma", Font.PLAIN, 13));
		GridBagConstraints gbc_lblNewLabel_1_1_3 = new GridBagConstraints();
		gbc_lblNewLabel_1_1_3.anchor = GridBagConstraints.EAST;
		gbc_lblNewLabel_1_1_3.fill = GridBagConstraints.VERTICAL;
		gbc_lblNewLabel_1_1_3.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_1_1_3.gridx = 0;
		gbc_lblNewLabel_1_1_3.gridy = 4;
		panel.add(lblNewLabel_1_1_3, gbc_lblNewLabel_1_1_3);
		
		textField_ChatBeo = new JTextField();
		textField_ChatBeo.setFont(new Font("Tahoma", Font.PLAIN, 13));
		textField_ChatBeo.setColumns(10);
		GridBagConstraints gbc_textField_ChatBeo = new GridBagConstraints();
		gbc_textField_ChatBeo.insets = new Insets(0, 0, 5, 0);
		gbc_textField_ChatBeo.fill = GridBagConstraints.HORIZONTAL;
		gbc_textField_ChatBeo.gridx = 2;
		gbc_textField_ChatBeo.gridy = 4;
		panel.add(textField_ChatBeo, gbc_textField_ChatBeo);
		
		JLabel lblNewLabel_1_1_4 = new JLabel("Độ trong");
		lblNewLabel_1_1_4.setFont(new Font("Tahoma", Font.PLAIN, 13));
		GridBagConstraints gbc_lblNewLabel_1_1_4 = new GridBagConstraints();
		gbc_lblNewLabel_1_1_4.anchor = GridBagConstraints.EAST;
		gbc_lblNewLabel_1_1_4.fill = GridBagConstraints.VERTICAL;
		gbc_lblNewLabel_1_1_4.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_1_1_4.gridx = 0;
		gbc_lblNewLabel_1_1_4.gridy = 5;
		panel.add(lblNewLabel_1_1_4, gbc_lblNewLabel_1_1_4);
		
		textField_DoTrong = new JTextField();
		textField_DoTrong.setFont(new Font("Tahoma", Font.PLAIN, 13));
		textField_DoTrong.setColumns(10);
		GridBagConstraints gbc_textField_DoTrong = new GridBagConstraints();
		gbc_textField_DoTrong.insets = new Insets(0, 0, 5, 0);
		gbc_textField_DoTrong.fill = GridBagConstraints.HORIZONTAL;
		gbc_textField_DoTrong.gridx = 2;
		gbc_textField_DoTrong.gridy = 5;
		panel.add(textField_DoTrong, gbc_textField_DoTrong);
		
		JLabel lblNewLabel_1_1_5 = new JLabel("Màu");
		lblNewLabel_1_1_5.setFont(new Font("Tahoma", Font.PLAIN, 13));
		GridBagConstraints gbc_lblNewLabel_1_1_5 = new GridBagConstraints();
		gbc_lblNewLabel_1_1_5.anchor = GridBagConstraints.EAST;
		gbc_lblNewLabel_1_1_5.fill = GridBagConstraints.VERTICAL;
		gbc_lblNewLabel_1_1_5.insets = new Insets(0, 0, 5, 5);
		gbc_lblNewLabel_1_1_5.gridx = 0;
		gbc_lblNewLabel_1_1_5.gridy = 6;
		panel.add(lblNewLabel_1_1_5, gbc_lblNewLabel_1_1_5);
		
		textField_Mau = new JTextField();
		textField_Mau.setFont(new Font("Tahoma", Font.PLAIN, 13));
		textField_Mau.setColumns(10);
		GridBagConstraints gbc_textField_Mau = new GridBagConstraints();
		gbc_textField_Mau.insets = new Insets(0, 0, 5, 0);
		gbc_textField_Mau.fill = GridBagConstraints.HORIZONTAL;
		gbc_textField_Mau.gridx = 2;
		gbc_textField_Mau.gridy = 6;
		panel.add(textField_Mau, gbc_textField_Mau);
		
		JLabel lblNewLabel_2 = new JLabel("   ");
		GridBagConstraints gbc_lblNewLabel_2 = new GridBagConstraints();
		gbc_lblNewLabel_2.fill = GridBagConstraints.BOTH;
		gbc_lblNewLabel_2.insets = new Insets(0, 0, 5, 0);
		gbc_lblNewLabel_2.gridx = 2;
		gbc_lblNewLabel_2.gridy = 7;
		panel.add(lblNewLabel_2, gbc_lblNewLabel_2);
		
		
		
		JPanel panel_1 = new JPanel();
		panel_1.setBorder(new TitledBorder(null, "K\u1EBFt qu\u1EA3 ph\u00E2n c\u1EE5m", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		panel_1.setBounds(417, 78, 239, 97);
		contentPane.add(panel_1);
		panel_1.setLayout(null);
		
		JLabel lblNewLabel_3 = new JLabel("Cụm");
		lblNewLabel_3.setFont(new Font("Tahoma", Font.PLAIN, 13));
		lblNewLabel_3.setBounds(104, 29, 30, 13);
		panel_1.add(lblNewLabel_3);
		
		lbl_KQ = new JLabel(". . .");
		lbl_KQ.setHorizontalAlignment(SwingConstants.CENTER);
		lbl_KQ.setFont(new Font("Tahoma", Font.BOLD, 13));
		lbl_KQ.setBounds(107, 59, 24, 13);
		panel_1.add(lbl_KQ);
	}
}
