from datetime import datetime, timedelta

DEFAULT_IMAGES = [
    'uploads/256de136-d5c8-41f2-8044-2f2b523100d21.png',
    'uploads/ChatGPT Image 01_17_50 21 thg 5, 20266.png',
]

AUTHORS = ['Ban biên tập', 'Nguyễn Linh', 'Mai Anh', 'Minh Khoa']


def build_content(title, intro, impact, actions, closing):
    action_items = ''.join(f'<li>{item}</li>' for item in actions)
    return (
        f'<p>{intro}</p>'
        f'<p>{impact}</p>'
        f'<ul>{action_items}</ul>'
        f'<p>{closing}</p>'
    )


def make_item(title, summary, content, category, author, created_at, is_featured=False, image_index=0, views=0):
    return {
        'title': title,
        'summary': summary,
        'content': content,
        'image': DEFAULT_IMAGES[image_index % len(DEFAULT_IMAGES)],
        'category': category,
        'author': author,
        'created_at': created_at,
        'updated_at': created_at,
        'views': views,
        'is_featured': is_featured,
        'status': 'published',
    }


def build_seed_articles():
    articles = []
    base_date = datetime(2026, 5, 1, 8, 30)

    environment_topics = [
        ('Việt Nam tăng tốc phân loại rác tại nguồn ở đô thị', 'Phân loại rác tại nguồn đang trở thành ưu tiên trong nhiều đô thị lớn của Việt Nam.', 'Việc tách rác hữu cơ, rác tái chế và rác còn lại giúp giảm tải cho hệ thống thu gom.', ['Mỗi gia đình nên có ít nhất 3 ngăn phân loại.', 'Chất thải hữu cơ có thể ủ thành phân bón.', 'Nhựa sạch và giấy khô cần để riêng để tái chế hiệu quả.'], 'Hành động nhỏ tại hộ gia đình có thể tạo ra thay đổi lớn cho môi trường đô thị.'),
        ('Kinh tế tuần hoàn mở đường cho tăng trưởng xanh', 'Kinh tế tuần hoàn giúp kéo dài vòng đời sản phẩm, giảm chất thải và tối ưu tài nguyên.', 'Mô hình này đặc biệt phù hợp với các ngành sản xuất, bao bì và tiêu dùng nhanh.', ['Thiết kế sản phẩm dễ sửa chữa.', 'Thu hồi nguyên liệu đầu vào sau sử dụng.', 'Ưu tiên vật liệu tái chế trong sản xuất.'], 'Các doanh nghiệp chuyển sang kinh tế tuần hoàn sẽ có lợi thế cạnh tranh bền vững.'),
        ('Rác nhựa đại dương và bài toán giảm phát thải', 'Rác nhựa đang là áp lực lớn với đại dương, hệ sinh thái và chuỗi thực phẩm.', 'Giảm nhựa dùng một lần là một trong những biện pháp cấp thiết nhất hiện nay.', ['Dùng bình nước và hộp đựng cá nhân.', 'Hạn chế ống hút và túi nilon.', 'Tăng tỷ lệ thu hồi nhựa sau tiêu dùng.'], 'Giảm nhựa từ đầu nguồn luôn hiệu quả hơn xử lý khi rác đã ra môi trường.'),
        ('Đô thị xanh cần hệ thống thu gom rác thông minh', 'Các thành phố đang thử nghiệm cảm biến và tối ưu tuyến thu gom rác theo thời gian thực.', 'Công nghệ giúp giảm nhiên liệu, giảm mùi và tăng hiệu quả vận hành.', ['Gắn cảm biến mực đầy trên thùng rác.', 'Tối ưu lộ trình xe thu gom.', 'Kết nối dữ liệu với trung tâm điều hành đô thị.'], 'Quản lý rác bằng dữ liệu đang trở thành xu hướng của đô thị thông minh.'),
        ('Tái chế thủy tinh giúp tiết kiệm năng lượng và tài nguyên', 'Thủy tinh có thể tái chế nhiều lần nếu được thu hồi và phân loại đúng cách.', 'Tái chế thủy tinh tiết kiệm năng lượng hơn so với sản xuất từ nguyên liệu thô.', ['Làm sạch chai lọ trước khi bỏ vào thùng tái chế.', 'Phân loại theo màu nếu địa phương yêu cầu.', 'Khuyến khích điểm thu hồi tại siêu thị và khu dân cư.'], 'Một dòng vật liệu tưởng như rất cũ có thể tiếp tục phục vụ nhiều vòng đời mới.'),
        ('Chính sách xanh thúc đẩy giảm rác thải bao bì', 'Nhiều địa phương tăng cường quy định với bao bì khó tái chế và nhựa dùng một lần.', 'Đây là bước quan trọng để chuyển dịch sang tiêu dùng bền vững.', ['Khuyến khích bao bì tái sử dụng.', 'Tăng phí với sản phẩm khó tái chế.', 'Ưu tiên nhà cung cấp có cam kết môi trường.'], 'Chính sách đúng đắn sẽ tạo lực đẩy cho thị trường xanh.'),
        ('Nhà ở xanh và thói quen giảm chất thải sinh hoạt', 'Thiết kế nhà ở xanh không chỉ tiết kiệm điện nước mà còn hỗ trợ giảm rác thải.', 'Không gian lưu trữ và phân loại rác tiện lợi giúp duy trì thói quen tốt.', ['Bố trí khu phân loại gần bếp.', 'Dùng thùng rác có nắp đậy.', 'Giảm mua hàng đóng gói quá mức.'], 'Môi trường sống tốt bắt đầu từ những thói quen rất gần gũi.'),
        ('Rác hữu cơ có thể biến thành nguồn phân bón hữu ích', 'Rác hữu cơ chiếm tỷ lệ lớn trong rác sinh hoạt và hoàn toàn có thể tận dụng.', 'Ủ rác đúng cách giúp giảm mùi, giảm phát thải và bổ sung dinh dưỡng cho đất.', ['Tách riêng rau củ, lá cây và thức ăn thừa.', 'Sử dụng chế phẩm ủ sinh học.', 'Theo dõi độ ẩm để ủ hiệu quả.'], 'Biến rác thành tài nguyên là cách tiếp cận bền vững nhất.'),
        ('Giảm ô nhiễm nhựa bắt đầu từ hành vi tiêu dùng', 'Thói quen mua sắm có ảnh hưởng trực tiếp đến lượng rác thải nhựa phát sinh.', 'Người tiêu dùng càng chọn sản phẩm bền vững, thị trường càng thay đổi nhanh hơn.', ['Ưu tiên sản phẩm nạp lại.', 'Chọn hàng bán theo khối lượng thay vì từng gói nhỏ.', 'Mang túi cá nhân khi đi chợ.'], 'Mỗi lựa chọn tại quầy hàng đều là một phiếu bầu cho tương lai môi trường.'),
        ('Tái chế giấy và bài toán rừng bền vững', 'Giấy tái chế góp phần giảm áp lực lên tài nguyên rừng và giảm năng lượng sản xuất.', 'Tách giấy sạch khỏi rác ướt là yếu tố quyết định hiệu quả tái chế.', ['Không làm bẩn giấy bằng thức ăn hoặc dầu mỡ.', 'Phân loại thùng giấy, báo, sách cũ riêng.', 'Khuyến khích văn phòng số hóa tài liệu.'], 'Tiết kiệm giấy hôm nay là bảo vệ rừng cho ngày mai.'),
        ('Sân chơi xanh trong trường học giúp nuôi dưỡng ý thức môi trường', 'Các trường học đang đưa hoạt động phân loại rác vào chương trình ngoại khóa.', 'Học sinh tiếp cận sớm với thói quen xanh sẽ giữ thói quen này lâu dài.', ['Đặt thùng phân loại tại hành lang.', 'Tổ chức ngày đổi rác lấy cây.', 'Lồng ghép bài học tái chế vào sinh hoạt lớp.'], 'Giáo dục môi trường là một khoản đầu tư dài hạn cho xã hội.'),
        ('Rác điện tử cần được thu hồi đúng quy trình', 'Thiết bị điện tử hỏng chứa vật liệu quý và cả thành phần nguy hại.', 'Thu hồi đúng cách giúp bảo vệ sức khỏe và tận dụng nguyên vật liệu.', ['Không bỏ pin và linh kiện lẫn rác sinh hoạt.', 'Mang thiết bị hỏng đến điểm thu gom riêng.', 'Xóa dữ liệu trước khi thanh lý thiết bị cũ.'], 'Rác điện tử là một nguồn tài nguyên nếu được xử lý đúng chuẩn.'),
        ('Không gian công cộng sạch hơn nhờ phân loại tại nguồn', 'Công viên, chợ và khu dân cư đều hưởng lợi khi rác được phân loại ngay từ đầu.', 'Chi phí vệ sinh giảm đáng kể nếu lượng rác lẫn tạp chất ít hơn.', ['Đặt hướng dẫn phân loại trực quan.', 'Tăng số thùng rác theo từng loại.', 'Tổ chức đội tình nguyện nhắc nhở tại điểm công cộng.'], 'Một thành phố sạch phụ thuộc vào từng thói quen nhỏ của cư dân.'),
        ('Tái chế kim loại giúp giảm khai thác quặng', 'Kim loại là vật liệu có khả năng tái chế cao và mang lại giá trị kinh tế lớn.', 'Thu hồi lon nhôm, sắt thép và vật dụng kim loại giúp tiết kiệm tài nguyên.', ['Tách kim loại khỏi rác còn ướt.', 'Ép gọn lon nhôm trước khi gom.', 'Ưu tiên thu mua ở các điểm tái chế uy tín.'], 'Mỗi kilogram kim loại tái chế đều giảm áp lực lên khai thác mới.'),
        ('Môi trường sạch hơn khi các khu dân cư chung tay', 'Các chương trình cộng đồng tạo ra tác động lớn hơn nhiều so với hành động riêng lẻ.', 'Khi mọi hộ gia đình đều tham gia, hệ thống thu gom sẽ vận hành trơn tru hơn.', ['Tổ chức ngày ra quân dọn rác.', 'Thiết lập nhóm nhắc lịch thu gom.', 'Tuyên dương khu phố làm tốt phân loại.'], 'Cộng đồng là nền tảng quan trọng nhất cho một môi trường bền vững.')
    ]

    waste_topics = [
        ('Cách nhận biết rác tái chế trong gia đình', 'Rác tái chế gồm những vật liệu có thể thu hồi và đưa vào vòng đời mới.', 'Phân loại đúng từ bếp và phòng khách là cách đơn giản nhất để tăng tỷ lệ tái chế.', ['Giấy sạch, chai nhựa và lon kim loại thường có thể tái chế.', 'Giữ rác tái chế khô để tránh giảm chất lượng vật liệu.', 'Kiểm tra hướng dẫn phân loại của địa phương.'], 'Khi rác được tách đúng, giá trị tái chế tăng rõ rệt.'),
        ('Ba ngăn phân loại rác cho mọi gia đình', 'Một hệ thống 3 ngăn giúp việc phân loại trở nên dễ nhớ và dễ duy trì.', 'Mô hình này phù hợp với hầu hết căn hộ và nhà phố.', ['Ngăn hữu cơ cho thức ăn thừa.', 'Ngăn tái chế cho giấy, nhựa, kim loại.', 'Ngăn còn lại cho rác không tái chế.'], 'Chìa khóa là sự nhất quán mỗi ngày.'),
        ('Mẹo xử lý rác hữu cơ không gây mùi', 'Rác hữu cơ cần được quản lý nhanh để tránh mùi và ruồi muỗi.', 'Nếu xử lý đúng, rác hữu cơ còn có thể tạo ra nguồn dinh dưỡng hữu ích.', ['Đậy kín thùng rác hữu cơ.', 'Đổ rác thường xuyên trong ngày nóng.', 'Trộn thêm vật liệu khô khi ủ compost.'], 'Rác hữu cơ có thể rất sạch nếu được quản lý chủ động.'),
        ('Rác thải sinh hoạt và cách giảm khối lượng mỗi ngày', 'Giảm rác sinh hoạt bắt đầu từ việc mua ít hơn và dùng lâu hơn.', 'Những thay đổi nhỏ sẽ giảm đáng kể khối lượng cần xử lý.', ['Ưu tiên đồ dùng bền và tái sử dụng.', 'Tách bao bì ra khỏi phần thực phẩm.', 'Hạn chế mua đồ dùng một lần.'], 'Ít rác hơn nghĩa là ít áp lực hơn lên hệ thống xử lý.'),
        ('Giấy bìa carton có thể tái chế như thế nào', 'Thùng carton, hộp giấy và bao bì giấy thường có giá trị thu hồi tốt.', 'Điều quan trọng là giữ giấy khô và sạch dầu mỡ.', ['Gập phẳng thùng carton trước khi bỏ.', 'Không trộn giấy với rác ướt.', 'Bỏ ghim kim loại nếu có thể.'], 'Thu gom đúng cách sẽ giúp giấy quay lại dây chuyền sản xuất.'),
        ('Nhựa PET, HDPE và những điều cần biết khi phân loại', 'Không phải loại nhựa nào cũng có quy trình tái chế giống nhau.', 'Nhận biết mã nhựa sẽ giúp người dùng bỏ đúng thùng và tăng hiệu quả phân loại.', ['Quan sát ký hiệu dưới đáy sản phẩm.', 'Rửa sạch chai lọ trước khi bỏ.', 'Tách nắp, nhãn nếu địa phương yêu cầu.'], 'Phân loại nhựa chính xác là bước quan trọng của kinh tế tuần hoàn.'),
        ('Tại sao rác ướt làm hỏng giá trị tái chế?', 'Một lượng nhỏ rác ướt cũng có thể làm giảm chất lượng cả mẻ vật liệu.', 'Đó là lý do nhiều địa phương nhấn mạnh việc giữ khô rác tái chế.', ['Không để thức ăn dính vào giấy.', 'Làm sạch hộp nhựa trước khi bỏ.', 'Dùng túi riêng cho rác tái chế khô.'], 'Giữ khô rác là thói quen đơn giản nhưng rất hiệu quả.'),
        ('Lịch thu gom rác giúp khu phố sạch và gọn', 'Khi cư dân nắm lịch thu gom, rác sẽ không bị tồn đọng trên vỉa hè.', 'Tổ chức tốt giúp giảm mùi, giảm côn trùng và giảm chi phí vệ sinh.', ['Niêm yết lịch thu gom tại tòa nhà.', 'Nhắc nhở cư dân bỏ rác đúng giờ.', 'Theo dõi phản hồi để điều chỉnh tần suất.'], 'Một lịch trình rõ ràng giúp cả khu dân cư vận hành tốt hơn.'),
        ('Thu gom pin cũ và rác nguy hại đúng chỗ', 'Pin, bóng đèn và một số hóa chất không nên bỏ chung với rác sinh hoạt.', 'Các điểm thu gom chuyên biệt giúp bảo vệ môi trường và an toàn xử lý.', ['Tách pin cũ vào hộp riêng.', 'Không đốt hoặc đập pin.', 'Đem đến điểm thu gom nguy hại gần nhất.'], 'Rác nguy hại cần cách xử lý hoàn toàn khác rác thường.'),
        ('Rác cồng kềnh nên xử lý ra sao?', 'Giường, tủ, nệm và đồ gia dụng cũ cần tuyến xử lý riêng.', 'Nếu không gom riêng, chúng dễ làm quá tải hệ thống thu gom.', ['Liên hệ đơn vị thu gom lớn.', 'Tháo rời nếu vật liệu có thể tái chế.', 'Tái sử dụng hoặc quyên góp khi còn tốt.'], 'Phân loại rác cồng kềnh giúp thành phố vận hành hiệu quả hơn.'),
        ('Làm sạch bao bì trước khi tái chế', 'Bao bì có thức ăn bám lại thường khó tái chế hoặc phải xử lý bổ sung.', 'Rửa nhanh bằng nước sạch có thể tạo khác biệt lớn.', ['Tráng hộp sữa, hộp nhựa trước khi bỏ.', 'Lau khô hộp dầu ăn hoặc hộp thực phẩm.', 'Không đổ lẫn thức ăn thừa vào thùng tái chế.'], 'Vài giây làm sạch có thể giúp vật liệu có thêm nhiều vòng đời.'),
        ('Giảm rác nhựa từ nhà bếp đến phòng tắm', 'Nhiều sản phẩm nhựa phát sinh mỗi ngày từ các khu vực rất nhỏ trong nhà.', 'Nếu thay đổi từng khu vực, lượng rác nhựa giảm đáng kể.', ['Chuyển sang chai đựng có thể nạp lại.', 'Dùng đồ bếp bền hơn đồ dùng một lần.', 'Ưu tiên sản phẩm không có bao bì quá mức.'], 'Giảm nhựa ở từng góc nhà là cách dễ bắt đầu nhất.'),
        ('Rác thải thực phẩm và bài toán thất thoát', 'Thực phẩm bỏ đi là một phần lớn của rác sinh hoạt và gây lãng phí tài nguyên.', 'Nghiên cứu thói quen mua sắm giúp giảm rất nhiều rác thực phẩm.', ['Lên kế hoạch bữa ăn trước khi mua.', 'Bảo quản đúng nhiệt độ.', 'Ưu tiên dùng nguyên liệu sắp hết hạn trước.'], 'Giảm lãng phí thực phẩm cũng là một dạng phân loại rác thông minh.'),
        ('Thói quen phân loại rác cho trẻ em', 'Trẻ em học rất nhanh nếu nhìn thấy hành vi tốt lặp lại hằng ngày.', 'Gia đình là môi trường đầu tiên để xây dựng ý thức phân loại.', ['Dán nhãn thùng rác trực quan.', 'Cho trẻ tham gia bỏ rác đúng chỗ.', 'Khen ngợi khi trẻ làm đúng.'], 'Giáo dục sớm tạo ra tác động lâu dài cho cộng đồng.'),
        ('Lưu ý khi phân loại thủy tinh vỡ và vật sắc nhọn', 'Thủy tinh vỡ cần được xử lý cẩn thận để tránh gây thương tích.', 'Việc đóng gói đúng cách giúp an toàn cho người thu gom.', ['Bọc giấy hoặc carton quanh vật sắc nhọn.', 'Dán nhãn cảnh báo lên túi rác.', 'Không để lẫn với giấy hoặc nhựa tái chế.'], 'An toàn cho người thu gom cũng là một phần của phân loại đúng.'),
    ]

    ai_topics = [
        ('AI giúp phân loại rác tại nguồn chính xác hơn', 'Mô hình trí tuệ nhân tạo đang hỗ trợ nhận diện rác nhanh và ổn định hơn.', 'Khi kết hợp camera và dữ liệu huấn luyện tốt, hệ thống có thể giảm sai sót đáng kể.', ['Thu thập ảnh rác theo nhiều góc độ.', 'Tăng cường dữ liệu bằng biến đổi ảnh.', 'Kết nối AI với hệ thống quản lý rác thông minh.'], 'AI không thay thế con người hoàn toàn mà giúp quy trình trở nên hiệu quả hơn.'),
        ('Robot phân loại rác tự động trong nhà máy tái chế', 'Robot công nghiệp đang được dùng để nhận diện và tách vật liệu trên băng chuyền.', 'Công nghệ này giảm chi phí lao động và tăng độ ổn định khi vận hành.', ['Dùng camera độ phân giải cao.', 'Kết hợp cảm biến và cánh tay robot.', 'Tối ưu thuật toán phân lớp theo từng vật liệu.'], 'Tự động hóa là chìa khóa để nâng tỷ lệ tái chế trong tương lai.'),
        ('Mạng nơ-ron hỗ trợ nhận diện vật liệu tái chế', 'Các mô hình học sâu có thể học đặc trưng từ ảnh rác và bao bì.', 'Điều này giúp phân biệt nhanh giấy, nhựa, kim loại và thủy tinh.', ['Huấn luyện trên dữ liệu đa dạng.', 'Chuẩn hóa ảnh đầu vào.', 'Đánh giá theo từng nhóm vật liệu.'], 'Hiệu quả của AI phụ thuộc rất lớn vào chất lượng dữ liệu.'),
        ('AI trong quản lý thùng rác thông minh', 'Thùng rác thông minh có thể báo khi đầy và tự ghi nhận loại rác được bỏ vào.', 'Nhờ đó, đội thu gom có thể tối ưu lịch trình và tuyến đường.', ['Gắn cảm biến mức đầy.', 'Tích hợp camera nhận diện.', 'Đẩy dữ liệu về trung tâm điều hành.'], 'Khi dữ liệu tốt hơn, vận hành đô thị cũng xanh hơn.'),
        ('Học máy giúp dự đoán lượng rác phát sinh', 'Dự đoán rác thải hỗ trợ thành phố lên kế hoạch nhân lực và phương tiện.', 'Các mô hình có thể học theo mùa, khu vực và hành vi người dùng.', ['Theo dõi dữ liệu lịch sử.', 'Kết hợp thời tiết và sự kiện.', 'Dự báo theo khung giờ cao điểm.'], 'Dự báo tốt giúp giảm quá tải trong hệ thống xử lý.'),
        ('Computer Vision giảm sai sót trong phân loại rác', 'Thị giác máy tính cho phép hệ thống nhận dạng hình ảnh rác nhanh hơn thao tác thủ công.', 'Kết quả chính xác hơn khi ảnh đầu vào được chụp rõ và đa dạng.', ['Tối ưu ánh sáng khi chụp.', 'Bổ sung dữ liệu nhiễu thực tế.', 'Dùng mô hình nhẹ để chạy trên thiết bị biên.'], 'CV là một lớp hạ tầng quan trọng của các hệ thống môi trường hiện đại.'),
        ('AI hỗ trợ tái chế trong mô hình kinh tế tuần hoàn', 'Tự động hóa và AI giúp tăng tốc phân loại, thu hồi và tái sử dụng vật liệu.', 'Điều này tạo nền tảng cho chuỗi giá trị tuần hoàn khép kín.', ['Theo dõi vòng đời vật liệu.', 'Tối ưu phân loại theo chất liệu.', 'Kết nối dữ liệu giữa thu gom và tái chế.'], 'Kinh tế tuần hoàn cần cả công nghệ lẫn hành vi xã hội.'),
        ('Phân tích dữ liệu rác thải để tối ưu thu gom', 'AI có thể phát hiện khu vực phát sinh nhiều rác để ưu tiên nguồn lực.', 'Cách tiếp cận này giúp giảm chi phí và tăng hiệu quả phục vụ.', ['Tích hợp bản đồ nhiệt.', 'Nhìn vào xu hướng phát sinh rác.', 'Điều chỉnh điểm đặt thùng rác theo dữ liệu.'], 'Quản trị dựa trên dữ liệu là tương lai của xử lý rác đô thị.'),
        ('Thiết bị biên chạy mô hình AI tại điểm thu gom', 'Edge AI cho phép xử lý tại chỗ mà không cần gửi toàn bộ dữ liệu lên cloud.', 'Cách làm này tiết kiệm băng thông và cải thiện thời gian phản hồi.', ['Dùng model tối ưu hóa kích thước.', 'Lưu tạm kết quả nhận diện cục bộ.', 'Đồng bộ dữ liệu theo chu kỳ.'], 'Xử lý gần nguồn là một chiến lược đáng giá cho đô thị thông minh.'),
        ('AI và bài toán minh bạch dữ liệu môi trường', 'Khi dữ liệu thu gom và phân loại được số hóa, các chỉ số môi trường trở nên rõ ràng hơn.', 'AI giúp biến dữ liệu thô thành báo cáo dễ hiểu cho người quản lý.', ['Chuẩn hóa định dạng ghi nhận.', 'Tạo dashboard theo thời gian thực.', 'Cảnh báo sớm khi số liệu bất thường.'], 'Minh bạch dữ liệu là nền tảng để cải thiện chính sách môi trường.')
    ]

    all_groups = [
        ('Môi trường', environment_topics),
        ('Phân loại rác', waste_topics),
        ('AI và xử lý rác', ai_topics),
    ]

    day_offset = 0
    for group_index, (category, topics) in enumerate(all_groups):
        for index, topic in enumerate(topics):
            if len(topic) == 6:
                title, summary, intro, impact, actions, closing = topic
            elif len(topic) == 5:
                title, summary, intro, actions, closing = topic
                impact = summary
            else:
                raise ValueError(f'Unexpected seed topic format: {topic!r}')
            created_at = base_date + timedelta(days=day_offset)
            articles.append(make_item(
                title=title,
                summary=summary,
                content=build_content(title, intro, impact, actions, closing),
                category=category,
                author=AUTHORS[(group_index + index) % len(AUTHORS)],
                created_at=created_at,
                is_featured=(group_index == 0 and index < 2) or (group_index == 1 and index == 0) or (group_index == 2 and index == 0),
                image_index=index + group_index,
                views=200 + (group_index * 50) + (index * 13),
            ))
            day_offset += 1

    return articles
