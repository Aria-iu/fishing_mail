use actix_files as fs;
use actix_files::NamedFile;
use actix_multipart::Multipart;
use actix_web::HttpRequest;
use actix_web::{post, web, App, HttpResponse, HttpServer, Responder};
use futures_util::stream::StreamExt as _;
use futures_util::TryStreamExt;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

async fn index(req: HttpRequest) -> actix_web::Result<NamedFile> {
    let path: PathBuf = req.match_info().query("filename").parse().unwrap();
    Ok(NamedFile::open(path)?)
}

#[post("/search")]
async fn search(mut payload: Multipart) -> impl Responder {
    // Iterate over multipart stream
    // 保存文件路径
    let mut file_path = String::new();
    while let Ok(Some(mut field)) = payload.try_next().await {
        let content_disposition = field.content_disposition().unwrap();
        let filename = content_disposition.get_filename().unwrap();

        // Create a file path to save the uploaded file
        file_path = format!("./uploads/{}", sanitize_filename::sanitize(&filename));

        // File object to save the file
        let mut f = std::fs::File::create(file_path.clone()).unwrap();

        // Save the file in chunks
        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            /*
            if let Ok(text) = std::str::from_utf8(&data) {
                println!("{}", text); // 打印出字符串
            } else {
                println!("{:?}", data); // 如果不是有效的字符串，就以字节形式打印出来
            }
             */
            f.write_all(&data).unwrap();
        }
    }

    // 调用Python脚本，并传递文件路径作为参数
    let output = Command::new("python3")
        .arg("main.py")
        .arg(&file_path)
        .output()
        .expect("Failed to execute Python script");

    // 获取并打印Python脚本的输出
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Python script output: {}", stdout);

    // 返回Python脚本的输出作为HTTP响应
    HttpResponse::Ok().body(stdout.into_owned())

    // HttpResponse::Ok().body("File uploaded successfully")
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(
                fs::Files::new("/static", "./static")
                    .show_files_listing()
                    .use_last_modified(true),
            )
            .service(search)
            .route("/{filename:.*}", web::get().to(index))
    })
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}


/*

#[post("/echo")]
async fn echo(req_body: String) -> impl Responder {
    HttpResponse::Ok().body(req_body)
}

async fn manual_hello() -> impl Responder {
    HttpResponse::Ok().body("Hey there!")
}

#[get("/stream")]
async fn stream() -> HttpResponse {
    let body = once(ok::<_, Error>(web::Bytes::from_static(b"test")));

    HttpResponse::Ok()
        .content_type("application/json")
        .streaming(body)
 */
